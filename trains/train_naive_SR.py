import tensorflow as tf
import numpy as np
import os,sys
import math
import time
from PIL import Image
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)

from models.naive_SR import naive_SR
from utils import utils
from data import data
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_path','/home/cc/data/sr_data2/small_set','input data path')
flags.DEFINE_integer('iterations',30000,'number of iterations')
flags.DEFINE_integer('batch_size',16,'batch size')
flags.DEFINE_string('train_dir','../ckpt/naive_SR/','model save path')
flags.DEFINE_string('log_path', '../ckpt/naive_SR_log/', 'model log save path')
flags.DEFINE_string('data_output_path','data/Output_data','output data path')
flags.DEFINE_integer('verbose',100,'show performance per X iterations')
flags.DEFINE_float('learning_rate',0.001,'learning rate for training')
flags.DEFINE_float('learning_rate_decay_rate',0.6,'learning rate for training')
flags.DEFINE_integer('save',1000,'save every X iterations')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')
flags.DEFINE_boolean('with_padding',False,'model with padding or not')
flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')

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

    #ckpt=tf.train.get_checkpoint_state(ckpt_path)
    #if ckpt and ckpt.model_checkpoint_path:
    #    print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
    #    model.saver.restore(session, ckpt.model_checkpoint_path)
    #else:
    print('Creating model with fresh parameters.')
    session.run(tf.global_variables_initializer())

    return model

def train():
    ckpt_path=FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    dataset=data.Dataset(FLAGS.data_path)
    #target_batch,input_batch=dataset.get_batch(FLAGS.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model=create_model(ckpt_path,FLAGS.optimizer,sess)
        writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        itr_print=FLAGS.verbose
        itr_save=FLAGS.save
        #itr_save_image=800
        loss=0
        iterations=FLAGS.iterations
        #iterations=1
        time_start=time.time()
        sys.stdout.flush()
        for itr in xrange(iterations):
        #for itr in xrange(20):
            #if itr==0:
            input_batch,target_batch=dataset.next_batch(FLAGS.batch_size)
            epoch=dataset.get_epoch()
            learning_rate_decay=FLAGS.learning_rate_decay_rate**epoch
            test,training_loss=model.step(sess,input_batch.astype(np.float32),target_batch.astype(np.float32),learning_rate_decay,training=True)
            loss+=training_loss
            #print test
            if((itr%itr_save==0 and itr!=0) or itr==FLAGS.iterations-1):
                model.saver.save(sess,ckpt_path+'train',model.global_step)
            if(itr%itr_print==0 and itr!=0):
                #print 'Iteration:'+str(itr)+' test:'+str(test)
                print 'Iteration:'+str(itr)+' Average loss:'+str(loss/itr_print)
                loss=0
                #for qqq in xrange(len(test)):
                #    print qqq
                #    print test[qqq]
                #print test
                time_end=time.time()
                print 'itr:'+str(itr)+'time used:'+str(time_end-time_start)+'s'
                time_start=time_end
                sys.stdout.flush()
            if(itr==(FLAGS.iterations-1) and itr%itr_print!=0):
                print 'Iteration:'+str(itr)+' Average loss:'+str(loss/(itr%itr_print))
            # if(itr!=0 and itr%itr_save_image==0):
            #     for i in xrange(16):
            #         test1=test[i]
            #         out_bmp = Image.fromarray(np.transpose(test1.astype(np.uint8),(1,0,2)))
            #         out_bmp.save('../data/DIV2K/bmp_out_'+str(i)+'.bmp')
            #         test1=input_batch[i]
            #         test1+=128
            #         out_bmp = Image.fromarray(np.transpose(test1.astype(np.uint8),(1,0,2)))
            #         out_bmp.save('../data/DIV2K/bmp_in_'+str(i)+'.bmp')
            #     lr_filename='../data/DIV2K/0028x2.png'
            #     input_data=Image.open(lr_filename)
            #     input_data=np.array(input_data)
            #     out_bmp = Image.fromarray(input_data.astype(np.uint8))
            #     out_bmp.save('../data/DIV2K/input.bmp')
            #     input_data=np.expand_dims(input_data,axis=0)
            #     input_data-=128
            #     prediction,_=model.step(sess,input_data.astype(np.float32),input_data.astype(np.float32),training=False)
            #     prediction=prediction[0]
            #     out_bmp = Image.fromarray(prediction.astype(np.uint8))
            #     out_bmp.save('../data/DIV2K/prediction.bmp')
    writer.close()


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()

