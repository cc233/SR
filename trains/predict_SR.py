import tensorflow as tf
import numpy as np
import os,sys
import math
from PIL import Image
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(data_path)
from utils import utils
from models.SR import SR
from utils import utils
from skimage.measure import compare_ssim as SSIM
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('test_data_in_path','../data/test/2.bmp','input data path')
flags.DEFINE_string('test_bmp_out_path','../data/test/bmp_out2.bmp','output bmp path')
flags.DEFINE_string('test_bmp_c1c2_out_path','../data/test/bmp_c1c2_out2.bmp','output bmp c1c2 path')
flags.DEFINE_string('test_baseline_path','../data/test/baseline2.bmp','output baseline path')
flags.DEFINE_string('bicubic_path','../data/test/bicubic2.bmp','output bicubic path')
flags.DEFINE_string('original_data_path','../data/test/gt/2.bmp','original data')
#flags.DEFINE_integer('iterations',1000,'number of iterations')
#flags.DEFINE_integer('batch_size','32','batch size')
flags.DEFINE_string('train_dir','../ckpt/SR/','model save path')
#flags.DEFINE_string('data_output_path','data/Output_data','output data path')
#flags.DEFINE_integer('verbose',10,'show performance per X iterations')
flags.DEFINE_float('learning_rate','0.001','learning rate for training')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')
flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')
flags.DEFINE_boolean('with_padding',False,'model with padding or not')

#input an image dir
#output the array of the image
def get_data(path):
    pass
    im=Image.open(path)
    data=np.array(im)
    lr=np.zeros([1,512,512,3],dtype=np.float32)
    lr[0,:,:,:]=data
    return lr
#input_data:lr_image
#target_data:hr_image
#ckpt_path:model save path
#optimizer:optimizer
#session:session
def create_model(ckpt_path,optimizer,session):
    model=SR(
        hidden_size=FLAGS.hidden_size,
        bottleneck_size=FLAGS.bottleneck_size,
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        dtype=tf.float32,
        scope='SR',
        scale=FLAGS.scale
        )

    ckpt=tf.train.get_checkpoint_state(ckpt_path)
    wrong=False
    if ckpt and ckpt.model_checkpoint_path:
        session.run(tf.global_variables_initializer())
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('There is something wrong!')
        #session.run(tf.global_variables_initializer())
        wrong=True

    return model,wrong

def im2double(im):

    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def PSNR(img1, img2):
    """
    
    :param img1: numpy format for image
    :param img2: 
    :return: the psnr value of two images
    """
    img1 = im2double(img1)
    img2 = im2double(img2) 
    mse = np.sum((img1-img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    N = img1.shape[0] * img1.shape[1] * img1.shape[2]
    #print(N)
    return 10.0 * math.log10( N / mse)
def bmptobmp_c1c2(bmp_in):
    shape=bmp_in.shape
    c1c2=np.reshape(bmp_in,(shape[0]/4,4,shape[1]/4,4,3))
    c1c2=np.transpose(c1c2,(0,2,1,3,4))
    c1c2=np.reshape(c1c2,(shape[0]/4,shape[1]/4,16,3))
    c2=np.max(c1c2,axis=2)
    c1=np.min(c1c2,axis=2)
    c1_out=c1
    c2_out=c2
    c1=np.expand_dims(c1,axis=2)
    c1=np.expand_dims(c1,axis=2)
    c2=np.expand_dims(c2,axis=2)
    c2=np.expand_dims(c2,axis=2)
    #c1:[shape[0]/4,shape[1]/4,3]
    temp=np.ones((1,1,4,4,3),dtype=np.uint8)
    c1=c1*temp
    c2=c2*temp
    c1=np.transpose(c1,(0,2,1,3,4))
    c2=np.transpose(c2,(0,2,1,3,4))
    c1=np.reshape(c1,(shape[0],shape[1],3,1))
    c2=np.reshape(c2,(shape[0],shape[1],3,1))
    c3=c1*1/3+c2*2/3
    c4=c1*2/3+c2*1/3
    c=np.concatenate((c1,c2,c3,c4),axis=3)
    c=c.astype(np.uint8)
    #cal distance
    bmp_in=np.reshape(bmp_in,(shape[0],shape[1],shape[2],1))
    dis_c1=np.square(bmp_in-c1)
    dis_c2=np.square(bmp_in-c2)
    dis_c3=np.square(bmp_in-c3)
    dis_c4=np.square(bmp_in-c4)

    dis=np.concatenate((dis_c1,dis_c2,dis_c3,dis_c4),axis=3)
    index=np.argmin(dis,axis=3)
    index=np.reshape(index,(-1))
    c=np.reshape(c,(-1,4))
    temp=np.arange(index.shape[0])
    bmp_out=c[temp,index]
    bmp_out=np.reshape(bmp_out,shape)
    return bmp_out,c1_out,c2_out
def cast(image):
    a=255*np.ones(image.shape,np.int32)
    b=np.zeros(image.shape,np.int32)
    cast_image=np.minimum(a,image)
    cast_image=np.maximum(b,cast_image)
    cast_image=cast_image.astype(np.uint8)
    return cast_image
def train():
    ckpt_path=FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    data=get_data(FLAGS.test_data_in_path)
    data-=128
    if not FLAGS.with_padding:
        data=utils.padding(data)
    #data=utils.padding(data)
    #target_batch,input_batch=dataset.get_batch(flags.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model,wrong=create_model(ckpt_path,FLAGS.optimizer,sess)
        if(wrong==True):
            return
        #_,training_loss=model.step(sess,input_batch,target_batch,training=True)
        c1c2_prediction,bmp_prediction,_=model.step(sess,data,data,1,training=False)

    #bmp_prediction
    bmp_prediction=np.reshape(bmp_prediction,bmp_prediction.shape[1:4])
    #print c1c2_prediction.shape
    out_bmp = Image.fromarray(bmp_prediction)
    out_bmp.save(FLAGS.test_bmp_out_path)

    #c1c2_prediction
    c1c2_prediction=np.reshape(c1c2_prediction,c1c2_prediction.shape[1:4])
    #print c1c2_prediction.shape
    out_bmp_c1c2 = Image.fromarray(c1c2_prediction)
    out_bmp_c1c2.save(FLAGS.test_bmp_c1c2_out_path)

    #evalute
    #bmp
    im=Image.open(FLAGS.original_data_path)
    original=np.array(im,dtype=np.uint8)
    PSNR_score=PSNR(original,bmp_prediction)
    SSIM_score,_=SSIM(original, bmp_prediction, full=True, multichannel=True)
    print 'BMP: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)

    #bmp_c1c2
    PSNR_score=PSNR(original,c1c2_prediction)
    SSIM_score,_=SSIM(original, c1c2_prediction, full=True, multichannel=True)
    print 'BMP_C1C2: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)

    #baseline
    bmp_bs,_,_=bmptobmp_c1c2(bmp_prediction.astype(np.int32))
    bmp_bs=cast(bmp_bs)
    PSNR_score=PSNR(original,bmp_bs)
    SSIM_score,_=SSIM(original, bmp_bs, full=True, multichannel=True)
    print 'Baseline: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)
    bmp_bs = Image.fromarray(bmp_bs)
    bmp_bs.save(FLAGS.test_baseline_path)

    #bicubic
    bicubic=Image.open(FLAGS.test_data_in_path)
    bicubic=bicubic.resize((bmp_prediction.shape[0],bmp_prediction.shape[1]), Image.BICUBIC)
    bicubic=np.array(bicubic,np.int32)
    bicubic,c1,c2=bmptobmp_c1c2(bicubic)
    bicubic=cast(bicubic)
    PSNR_score=PSNR(original,bicubic)
    SSIM_score,_=SSIM(original, bicubic, full=True, multichannel=True)
    print 'Bicubic: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)
    bicubic = Image.fromarray(bicubic)
    bicubic.save(FLAGS.bicubic_path)
def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()

