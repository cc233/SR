import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
import tensorflow as tf
class Dataset(object):

    def __init__(self, path, read_c1c2 = False):
        pass
        self.read_c1c2=read_c1c2
        if read_c1c2:
            self.lr_bmp,self.hr_bmp,self.lr_c1c2,self.hr_c1c2,self.num_train=self.read_data_with_c1c2(path)
        else:
            self.lr_bmp,self.hr_bmp,self.num_train=self.read_data(path)
        # I think here we should minus the mean value
        self.lr_bmp-=128
        self.epoch=0
        self.index_in_epoch=0
    def next_batch(self,batch_size):
        start=self.index_in_epoch
        self.index_in_epoch+=batch_size
        if self.index_in_epoch>self.num_train:
            self.epoch+=1
            perm=np.arange(self.num_train)
            np.random.shuffle(perm)
            if self.read_c1c2:
                self.lr_c1c2=self.lr_c1c2[perm]
                self.hr_c1c2=self.hr_c1c2[perm]
            self.lr_bmp=self.lr_bmp[perm]
            self.hr_bmp=self.hr_bmp[perm]
            start=0
            self.index_in_epoch=batch_size
        end=self.index_in_epoch
        if self.read_c1c2:
            return self.lr_bmp[start:end],self.hr_bmp[start:end],self.lr_c1c2[start:end],self.hr_c1c2[start:end]
        else:
            return self.lr_bmp[start:end],self.hr_bmp[start:end]
    def get_epoch(self):
        return self.epoch
    def read_data_with_c1c2(self,path):
        lr_bmp_path=os.path.join(path,'lr_bmp.npy')
        hr_bmp_path=os.path.join(path,'hr_bmp.npy')
        lr_c1c2_path=os.path.join(path,'lr_c1c2.npy')
        hr_c1c2_path=os.path.join(path,'hr_c1c2.npy')

        #load data
        lr_bmp=np.load(lr_bmp_path)
        hr_bmp=np.load(hr_bmp_path)
        lr_c1c2=np.load(lr_c1c2_path)
        hr_c1c2=np.load(hr_c1c2_path)

        #shuffle data
        perm=np.arange(lr_bmp.shape[0])
        np.random.shufle(perm)
        lr_bmp=lr_bmp[perm]
        hr_bmp=hr_bmp[perm]
        lr_c1c2=lr_c1c2[perm]
        hr_c1c2=hr_c1c2[perm]
        print('read data complete.')
        return lr_bmp,hr_bmp,lr_c1c2,hr_c1c2,lr_bmp.shape[0]

    def read_data(self,path):
        lr_bmp_path=os.path.join(path,'lr.npy')
        hr_bmp_path=os.path.join(path,'hr.npy')
        #print lr_path
        lr_bmp=np.load(lr_bmp_path)
        hr_bmp=np.load(hr_bmp_path)
        #shuffle data
        perm=np.arange(lr_bmp.shape[0])
        np.random.shuffle(perm)
        lr_bmp=lr_bmp[perm]
        hr_bmp=hr_bmp[perm]
        print('read data complete.')
        return lr_bmp,hr_bmp,lr_bmp.shape[0]

    #used for testing
    def get_data(self):
        return self.input_croped,self.target_croped
