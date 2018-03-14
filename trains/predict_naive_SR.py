import tensorflow as tf
import numpy as np
import os,sys
import math
from PIL import Image
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(data_path)
from utils import utils
from models.naive_SR import naive_SR
from utils import utils
from skimage.measure import compare_ssim as SSIM
flags=tf.app.flags
FLAGS=flags.FLAGS
#our dataset
flags.DEFINE_string('test_data_in_path','../data/test/5.bmp','input data path')
flags.DEFINE_string('test_bmp_out_path','../data/test/bmp_out5.bmp','output bmp path')
flags.DEFINE_string('bicubic_path','../data/test/bicubic5.bmp','output bicubic path')
flags.DEFINE_string('original_data_path','../data/test/gt/5.bmp','original data')

#DIV2K
# flags.DEFINE_string('test_data_in_path','../data/DIV2K/input/0003x2.png','input data path')
# flags.DEFINE_string('test_bmp_out_path','../data/DIV2K/bmp_out0001.bmp','output bmp path')
# flags.DEFINE_string('bicubic_path','../data/DIV2K/bicubic0028.bmp','output bicubic path')
# flags.DEFINE_string('original_data_path','../data/DIV2K/input/0003.png','original data')

#flags.DEFINE_integer('iterations',1000,'number of iterations')
#flags.DEFINE_integer('batch_size','32','batch size')
flags.DEFINE_string('train_dir','../ckpt/naive_SR/','model save path')
#flags.DEFINE_string('data_output_path','data/Output_data','output data path')
#flags.DEFINE_integer('verbose',10,'show performance per X iterations')
flags.DEFINE_float('learning_rate','0.001','learning rate for training')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')
flags.DEFINE_boolean('with_padding',False,'model with padding or not')
flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')

#input an image dir
#output the array of the image
def get_data(path):
    pass
    im=Image.open(path)
    data=np.array(im)
    #data=np.transpose(data,(1,0,2))
    lr=np.expand_dims(data,axis=0)
    return lr
#input_data:lr_image
#target_data:hr_image
#ckpt_path:model save path
#optimizer:optimizer
#session:session
def create_model(ckpt_path,optimizer,session):
    model=naive_SR(
        hidden_size=FLAGS.hidden_size,
        bottleneck_size=FLAGS.bottleneck_size,
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        dtype=tf.float32,
        scope='naive_SR',
        scale=FLAGS.scale,
        with_padding=FLAGS.with_padding
        )

    ckpt=tf.train.get_checkpoint_state(ckpt_path)
    wrong=False
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('There is something wrong!')
        #session.run(tf.global_variables_initializer())
        wrong=True

    return model,wrong

#get PSNR of 2 image
# def PSNR(img1, img2):
#     """
    
#     :param img1: numpy format for image
#     :param img2: 
#     :return: the psnr value of two images
#     """
#     mse = np.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
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

def get_bmp_c1c2_from_gt(bmp_in,c1,c2):
    shape=bmp_in.shape
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
    return bmp_out
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
    temp=np.ones((1,1,4,4,3),dtype=np.int32)
    c1=c1*temp
    c2=c2*temp
    c1=np.transpose(c1,(0,2,1,3,4))
    c2=np.transpose(c2,(0,2,1,3,4))
    c1=np.reshape(c1,(shape[0],shape[1],3,1))
    c2=np.reshape(c2,(shape[0],shape[1],3,1))
    c3=c1*1/3+c2*2/3
    c4=c1*2/3+c2*1/3
    c=np.concatenate((c1,c2,c3,c4),axis=3)
    c=c.astype(np.int32)
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
    bmp_out=bmp_out.astype(np.uint8)
    return bmp_out,c1_out,c2_out
def train():
    ckpt_path=FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    data=get_data(FLAGS.test_data_in_path)
    data=data.astype(np.float32)-128
    if not FLAGS.with_padding:
        data=utils.padding(data)
    #print data
    #target_batch,input_batch=dataset.get_batch(flags.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model,wrong=create_model(ckpt_path,FLAGS.optimizer,sess)
        if(wrong==True):
            return
        #_,training_loss=model.step(sess,input_batch,target_batch,training=True)
        print data.shape
        bmp_prediction,_=model.step(sess,data,data,1,training=False)
        print bmp_prediction.shape
    #bmp_prediction
    bmp_prediction=np.reshape(bmp_prediction.astype(np.uint8),bmp_prediction.shape[1:4])
    #print c1c2_prediction.shape
    out_bmp = Image.fromarray(bmp_prediction)
    out_bmp.save(FLAGS.test_bmp_out_path)


    #evalute
    #bmp
    im=Image.open(FLAGS.original_data_path)
    #im.save('test111.bmp')
    original=np.array(im,dtype=np.uint8)
    PSNR_score=PSNR(original,bmp_prediction)
    SSIM_score,_=SSIM(original, bmp_prediction, full=True, multichannel=True)
    print 'BMP: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)


    #bicubic
    bicubic=Image.open(FLAGS.test_data_in_path)
    bicubic=bicubic.resize((bmp_prediction.shape[1],bmp_prediction.shape[0]), Image.BICUBIC)
    bicubic=np.array(bicubic,dtype=np.uint8)

    #bicubic,_,_=bmptobmp_c1c2(bicubic)
    PSNR_score=PSNR(original,bicubic)
    SSIM_score,_=SSIM(original, bicubic, full=True, multichannel=True)
    print 'Bicubic: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)
    bicubic = Image.fromarray(bicubic)
    bicubic.save(FLAGS.bicubic_path)

    #c1c2
    c1c2_path='/home/chenchen/sr/bmp_c1c2_sr/data/test/gt'
    c1_path=os.path.join(c1c2_path,'5_c1.bmp')
    c2_path=os.path.join(c1c2_path,'5_c2.bmp')
    bmp_c1c2_path=os.path.join(c1c2_path,'5_c1c2.bmp')

    c1=Image.open(c1_path)
    c1=np.array(c1,dtype=np.int32)
    c2=Image.open(c2_path)
    c2=np.array(c2,dtype=np.int32)
    bmp_c1c2_gt=Image.open(bmp_c1c2_path)
    bmp_c1c2_gt=np.array(bmp_c1c2_gt,dtype=np.uint8)
    bmp_c1c2=get_bmp_c1c2_from_gt(bmp_prediction,c1,c2)
    bmp_c1c2_2,_,_=bmptobmp_c1c2(bmp_prediction)
    PSNR_score=PSNR(original,bmp_c1c2_gt)
    SSIM_score,_=SSIM(original, bmp_c1c2_gt, full=True, multichannel=True)
    print 'bmp_c1c2_gt: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)

    PSNR_score=PSNR(original,bmp_c1c2)
    SSIM_score,_=SSIM(original, bmp_c1c2, full=True, multichannel=True)
    print 'bmp_c1c2: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)
    bmp_c1c2 = Image.fromarray(bmp_c1c2)
    bmp_c1c2.save('../data/test/bmp_c1c2_out5.bmp')

    PSNR_score=PSNR(original,bmp_c1c2_2)
    SNR_score=PSNR(original,bmp_c1c2_2)
    SSIM_score,_=SSIM(original, bmp_c1c2_2, full=True, multichannel=True)
    print 'bmp_c1c2_2: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)
    bmp_c1c2_2 = Image.fromarray(bmp_c1c2_2)
    bmp_c1c2_2.save('../data/test/bmp_c1c2_2_out5.bmp')
    # #another method
    # another=Image.open(FLAGS.another_data_path)
    # another=np.array(another,dtype=np.int8)
    # another=np.transpose(another,(1,0,2))
    # #bicubic,_,_=bmptobmp_c1c2(bicubic)
    # another=np.uint8(another)
    # PSNR_score=PSNR(original,another)
    # SSIM_score,_=SSIM(original, another, full=True, multichannel=True)
    # print 'UIUC method: PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()

