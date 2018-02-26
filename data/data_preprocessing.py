from PIL import Image
import numpy as np
import os
import os.path
import tensorflow as tf
def crop(lr_in,hr_in):
    with tf.device('/cpu:0'):
        scale=2
        height=256
        width=256
        overlap=0
        hr=tf.placeholder(tf.float16,shape=[None,1024,1024,3])
        lr=tf.placeholder(tf.float16,shape=[None,512,512,3])
        #print 'hello'
        hr_crop = tf.extract_image_patches(hr, [1, height, width, 1], [1, height - 2 * overlap, width - 2 * overlap, 1], [1, 1, 1, 1], padding='VALID')
        hr_reshape=tf.reshape(hr_crop, [tf.shape(hr_crop)[0] * tf.shape(hr_crop)[1] * tf.shape(hr_crop)[2], height, width, 3])
        #print 'hi'
        lr_crop = tf.extract_image_patches(lr, [1, height/scale, width/scale, 1], [1, height/scale - 2 * overlap/scale, width/scale - 2 * overlap/scale, 1], [1, 1, 1, 1], padding='VALID')
        lr_reshape=tf.reshape(lr_crop, [tf.shape(lr_crop)[0] * tf.shape(lr_crop)[1] * tf.shape(lr_crop)[2], height/scale, width/scale, 3])
        sess = tf.Session()
        return sess.run([lr_reshape,hr_reshape],{lr:lr_in,hr:hr_in})

rootdir='/home/cc/data/sr_data2/small_set/'
lr_path=os.path.join(rootdir,'small_c1c2')
hr_path=os.path.join(rootdir,'big')
num=0
idx=0
for parent,dirnames,filenames in os.walk(lr_path):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        #print image_name
        num+=1
    print num
#hr=np.zeros(num*8,1000,1000,3)
lr=np.zeros([num*4,512,512,3],dtype=np.float16)
hr=np.zeros([num*4,1024,1024,3],dtype=np.float16)
#read_data
for parent,dirnames,filenames in os.walk(lr_path):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        hr_image_name=os.path.join(hr_path,filename)
        im=Image.open(image_name)
        hr_im=Image.open(hr_image_name)
        data=np.array(im)
        hr_data=np.array(hr_im)
        lr[idx,:,:,:]=data
        hr[idx,:,:,:]=hr_data
        idx+=1
        #print idx
        for i in xrange(3):
            data=np.rot90(data)
            hr_data=np.rot90(hr_data)
            lr[idx,:,:,:]=data
            hr[idx,:,:,:]=hr_data
            idx+=1
            #print idx
        
print idx
for i in xrange(lr.shape[0]):
    lr_test=lr[0,:,:,:]
    hr_test=hr[0,:,:,:]
    if np.sum(lr_test)<0.1 or np.sum(hr_test)<0.1:
        print 'something wrong: '+i
#crop data
lr_crop=None
hr_crop=None
for i in xrange(8):
    num=lr.shape[0]/8
    if(i==0):
        lr_crop,hr_crop=crop(lr[0:num],hr[0:num])
    else:
        lr_temp,hr_temp=crop(lr[i*num:(i+1)*num],hr[i*num:(i+1)*num])
        lr_crop=np.concatenate((lr_crop,lr_temp),axis=0)
        hr_crop=np.concatenate((hr_crop,hr_temp),axis=0)
    print i
np.save(os.path.join(rootdir,'lr'),lr_crop)
np.save(os.path.join(rootdir,'hr'),hr_crop)
