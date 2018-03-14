import tensorflow as tf
import numpy as np
import os,sys
import math
import time
from PIL import Image
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
#from tensorflow.python.tools import inspect_checkpoint as chkp
from models.SR import SR
from utils import utils
from data import data
from tensorflow.python import pywrap_tensorflow
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_path','/home/cc/data/sr_data2/small_set/','input data path')
flags.DEFINE_integer('iterations',30000,'number of iterations')
flags.DEFINE_integer('batch_size',1,'batch size')
flags.DEFINE_string('train_dir','../ckpt/SR/','model save path')
flags.DEFINE_string('log_path', '../ckpt/SR_log/', 'model log save path')
flags.DEFINE_integer('verbose',200,'show performance per X iterations')
flags.DEFINE_float('learning_rate',0.001,'learning rate for training')
flags.DEFINE_float('learning_rate_decay_rate',0.8,'learning rate for training')
flags.DEFINE_integer('save',1000,'save every X iterations')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')
flags.DEFINE_boolean('with_padding',False,'model with padding or not')
flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')

def get_checkpoint_var(ckpt):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    #print var_to_shape_map
    #print type(var_to_shape_map)
    not_restore=['SR/beta1_power','SR/beta2_power', 'SR/learning_rate','SR/global_step']
    var_to_restore={}
    for key in var_to_shape_map:
        var_to_restore[key]=reader.get_tensor(key)
    return var_to_restore
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
        scale=FLAGS.scale,
        with_padding=FLAGS.with_padding
        )

    ckpt=tf.train.get_checkpoint_state('../ckpt/SR_c1c2loss_0.001/')
    #ckpt=None
    #chkp.print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, tensor_name='', all_tensors=True)
    #var_to_restore=get_checkpoint_var(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        session.run(tf.global_variables_initializer())
        #model.saver=tf.train.Saver(var_to_restore)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Creating model with fresh parameters.')
        session.run(tf.global_variables_initializer())

    return model

def train():
    ckpt_path=FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    dataset=data.Dataset(FLAGS.data_path,True)
    #target_batch,input_batch=dataset.get_batch(FLAGS.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model=create_model(ckpt_path,FLAGS.optimizer,sess)
        writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        itr_print=FLAGS.verbose
        itr_save=FLAGS.save
        itr_save_image=600
        loss=0
        iterations=FLAGS.iterations
        #iterations=1
        time_start=time.time()
        sys.stdout.flush()
        test_dir='/home/chenchen/sr/bmp_c1c2_sr/data/test/'
        for itr in xrange(iterations):
        #for itr in xrange(20):
            #if itr==0:
            lr_bmp,hr_bmp,lr_c1c2,hr_c1c2=dataset.next_batch(FLAGS.batch_size)
            #print input_batch.shape
            #print target_batch.shape
            epoch=dataset.get_epoch()
            learning_rate_decay=FLAGS.learning_rate_decay_rate**epoch
            test,test_loss,test_target=model.step(
                        sess,
                        lr_bmp.astype(np.float32),
                        hr_bmp.astype(np.float32),
                        hr_c1c2.astype(np.float32),
                        learning_rate_decay,
                        training=True)
            #loss+=training_loss
            #print test
            if((itr%itr_save==0 and itr!=0) or itr==FLAGS.iterations-1):
                model.saver.save(sess,ckpt_path+'train',model.global_step)
            if(itr%itr_print==0 and itr!=0):
                #print 'Iteration:'+str(itr)+' test:'+str(test)
                print 'Iteration:'+str(itr)+' Average loss:'+str(loss/itr_print)
                loss=0
                #print test
                #for qqq in xrange(len(test)):
                #    print qqq
                #    print test[qqq]
                #print test
                time_end=time.time()
                print 'itr:'+str(itr)+'time used:'+str(time_end-time_start)+'s'
                time_start=time_end
                print test_loss
                sys.stdout.flush()
            if(itr==(FLAGS.iterations-1) and itr%itr_print!=0):
                print 'Iteration:'+str(itr)+' Average loss:'+str(loss/(itr%itr_print))
                print test_loss
            if(itr%itr_save_image==0 and itr<20000):
                bmp_prediction=test[0]
                c1c2_prediction=test[1]
                gt=test_target[0]
                bmp_prediction=bmp_prediction[0]
                c1c2_prediction=c1c2_prediction[0]
                gt=gt[0]
                print bmp_prediction.shape
                x=0
                y=208
                range1=4
                print 'bmp_prediction'
                print bmp_prediction[x:x+range1,y:y+range1,0]
                print '\n'
                print bmp_prediction[x:x+range1,y:y+range1,1]
                print '\n'
                print bmp_prediction[x:x+range1,y:y+range1,2]
                print '\n'
                print 'c1c2_prediction'
                print c1c2_prediction[x:x+range1,y:y+range1,0]
                print '\n'
                print c1c2_prediction[x:x+range1,y:y+range1,1]
                print '\n'
                print c1c2_prediction[x:x+range1,y:y+range1,2]
                print '\n'
                print 'gt'
                print gt[x:x+range1,y:y+range1,0]
                print '\n'
                print gt[x:x+range1,y:y+range1,1]
                print '\n'
                print gt[x:x+range1,y:y+range1,2]
                print '\n'
                test1_dir=os.path.join(test_dir,str(itr))
                if not os.path.exists(test1_dir):
                    os.makedirs(test1_dir)
                gt_filename='../data/test'
                lr_filename='../data/test/11.bmp'
                input_data=Image.open(lr_filename)
                input_data=np.array(input_data,dtype=np.float32)                
                input_data=np.expand_dims(input_data,axis=0)
                input_data-=128
                c1c2_prediction,bmp_prediction,test1=model.step(sess,input_data.astype(np.float32),input_data.astype(np.float32),1,training=False)
                c1c2_prediction=c1c2_prediction[0]
                bmp_prediction=bmp_prediction[0]
                out_bmp = Image.fromarray(bmp_prediction)
                out_bmp.save(os.path.join(test1_dir,'bmp_prediction1.bmp'))
                out_bmp = Image.fromarray(c1c2_prediction)
                out_bmp.save(os.path.join(test1_dir,'c1c2_prediction1.bmp'))
                avg_target=test1[2:5]
                avg_bmp=test1[5:8]
                avg_bmp_c1c2=test1[8:11]
                avg_c1=test1[11:14]
                avg_c2=test1[14:17]
                avg_c1c2_mean=test1[17:20]
                # print 'bmp_prediction1'
                # print 'avg_target:'
                # print avg_target
                # print 'avg_bmp:'
                # print avg_bmp
                # print 'avg_bmp_c1c2:'
                # print avg_bmp_c1c2
                # print 'avg_c1:'
                # print avg_c1
                # print 'avg_c2:'
                # print avg_c2
                # print 'avg_c1c2_mean:'
                # print avg_c1c2_mean
                # print '\n'
    writer.close()


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()

