import tensorflow as tf
from utils import utils

class SR(object):
    def __init__(self,
                 hidden_size,
                 bottleneck_size,
                 learning_rate,
                 optimizer='adam',
                 dtype=tf.float32,
                 scope='SR',
                 scale=2,
                 with_padding=False
                ):
        self.hidden_size=hidden_size
        self.bottleneck_size=bottleneck_size
        self.optimizer=optimizer
        self.dtype=dtype
        self.scale=scale
        self.with_padding=with_padding
        with tf.variable_scope(scope):
            self.init_learning_rate = tf.Variable(learning_rate, trainable=False, dtype=self.dtype, name='learning_rate')
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
            self.test=[]
            self.test1=[]
            self.test2=[]
            self.build_graph()

            self.saver = tf.train.Saver(max_to_keep=3)
        #for var in tf.global_variables():
        #    print var
    def build_graph(self):
        self._create_placeholder()
        self._create_SR_struct_without_padding()
        self._create_c1c2_struct()
        self._create_loss_without_padding()
        self._create_optimizer()
    
    def _create_placeholder(self):
        pass
        self.input=tf.placeholder(self.dtype,[None,None,None,3],name='input')
        self.target=tf.placeholder(self.dtype,[None,None,None,3],name='output')
        self.c1c2_gt=tf.placeholder(self.dtype,[None,None,None,6],name='c1c2_gt')
        self.learning_rate_decay=tf.placeholder(tf.float32,name='learning_rate_decay')
    def _create_SR_struct_without_padding(self):
        x=tf.layers.conv2d(self.input,self.hidden_size,1,activation=None,name='in')
        #self.test=tf.reduce_mean(x)
        #low resolution
        for i in range(6):
            x=utils.crop_by_pixel(x,1)+self.conv(x,self.hidden_size,self.bottleneck_size,'lr_conv'+str(i))
        temp=tf.nn.relu(x)
        #up sampling
        x=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2)+tf.layers.conv2d_transpose(temp,self.hidden_size,2,strides=2,name='up_sampling')

        #high resolution
        for i in range(4):
            x=utils.crop_by_pixel(x,1)+self.conv(x,self.hidden_size,self.bottleneck_size,'hr_conv'+str(i))
        x=tf.nn.relu(x)
        x=tf.layers.conv2d(x,3,1,name='out')
        bicubic=tf.image.resize_bicubic(self.input+128, tf.shape(self.input)[1:3] * 2, name='bicubic')
        bicubic=utils.crop_center(bicubic,tf.shape(x)[1:3])
        self.bmp_prediction=x+bicubic
        self.bmp_prediction_cast=tf.saturate_cast(self.bmp_prediction, tf.uint8)
        self.test1.append(self.bmp_prediction_cast)
    def _create_c1c2_struct(self):
        c=tf.layers.conv2d(self.bmp_prediction,self.hidden_size,1,activation=None,name='c1c2_in')
        #high resolution
        for i in range(3):
            c=self.conv(c,self.hidden_size,self.bottleneck_size,'hr_c_conv'+str(i),with_padding=True)
        temp=tf.nn.relu(c)
        #down sampling
        c=tf.layers.average_pooling2d(c,2,2,padding = 'same')+tf.layers.conv2d(temp,self.hidden_size,3,strides=2, padding = 'same',name='down_sampling1')
        
        #mid resolution
        for i in range(3):
            c=self.conv(c,self.hidden_size,self.bottleneck_size,'mr_c_conv'+str(i),with_padding=True)
        temp=tf.nn.relu(c)
        #down sampling
        c=tf.layers.average_pooling2d(c,2,2,padding = 'same')+tf.layers.conv2d(temp,self.hidden_size,3,strides=2, padding = 'same',name='down_sampling2')
        #low resolution
        for i in range(3):
            c=self.conv(c,self.hidden_size,self.bottleneck_size,'lr_c_conv'+str(i),with_padding=True)
        c=tf.nn.relu(c)
        c=tf.layers.conv2d(c,6,1,name='c1c2')
        #c=tf.cast(c,dtype=tf.int32)
        #get c1c2
        c1=c[:,:,:,0:3]
        c2=c[:,:,:,3:]
        self.c1=tf.minimum(c1,c2)
        self.c2=tf.maximum(c1,c2)
        #get c1c2_bmp
        self.c1c2_prediction=self._c1c2tobmp(self.c1,self.c2,self.bmp_prediction)
        self.c1c2_prediction_cast=tf.saturate_cast(self.c1c2_prediction, tf.uint8)
        self.test1.append(self.c1c2_prediction_cast)
         
        # avg0=tf.reduce_mean(self.target[:,:,:,0])
        # avg1=tf.reduce_mean(self.target[:,:,:,1])
        # avg2=tf.reduce_mean(self.target[:,:,:,2])
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
        # avg0=tf.reduce_mean(self.bmp_prediction[:,:,:,0])
        # avg1=tf.reduce_mean(self.bmp_prediction[:,:,:,1])
        # avg2=tf.reduce_mean(self.bmp_prediction[:,:,:,2])
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
        # avg0=tf.reduce_mean(self.c1c2_prediction[:,:,:,0])
        # avg1=tf.reduce_mean(self.c1c2_prediction[:,:,:,1])
        # avg2=tf.reduce_mean(self.c1c2_prediction[:,:,:,2])
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
        # avg0=tf.reduce_mean(self.c1[:,:,:,0])
        # avg1=tf.reduce_mean(self.c1[:,:,:,1])
        # avg2=tf.reduce_mean(self.c1[:,:,:,2])
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
        # avg0=tf.reduce_mean(self.c2[:,:,:,0])
        # avg1=tf.reduce_mean(self.c2[:,:,:,1])
        # avg2=tf.reduce_mean(self.c2[:,:,:,2])
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
        # avg0=tf.reduce_mean(self.c2[:,:,:,0]+self.c1[:,:,:,0])/2
        # avg1=tf.reduce_mean(self.c2[:,:,:,1]+self.c1[:,:,:,1])/2
        # avg2=tf.reduce_mean(self.c2[:,:,:,2]+self.c1[:,:,:,2])/2
        # self.test1.append(avg0)
        # self.test1.append(avg1)
        # self.test1.append(avg2)
    def _c1c2tobmp_fromc1c2gt(self,c1,c2,c1_gt,c2_gt,bmp_in):
        c1_shape=tf.shape(c1)
        #put 3 channels to batch_size 
        c1=tf.transpose(c1,(0,3,1,2))
        c2=tf.transpose(c2,(0,3,1,2))
        c1=tf.reshape(c1,[c1_shape[0]*3,c1_shape[1],c1_shape[2]])
        c2=tf.reshape(c2,[c1_shape[0]*3,c1_shape[1],c1_shape[2]])
        #c1,c2:[batch_size*3,h,w]

        #expand c1,c2 from batch_size*h*w to batch_size*4h*4w
        temp=tf.ones((1,1,1,4,4),dtype=tf.float32)
        c1=tf.expand_dims(c1,axis=3)
        c1=tf.expand_dims(c1,axis=4)
        c2=tf.expand_dims(c2,axis=3)
        c2=tf.expand_dims(c2,axis=4)
        c1=c1*temp
        c2=c2*temp
        c1=tf.reshape(tf.transpose(c1,(0,1,3,2,4)),[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2]])
        c2=tf.reshape(tf.transpose(c2,(0,1,3,2,4)),[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2]])
        c1=tf.expand_dims(c1,axis=3)
        c2=tf.expand_dims(c2,axis=3)
        #get c3,c4
        c3=tf.cast(c1*1/3+c2*2/3,dtype=tf.float32)
        c4=tf.cast(c1*2/3+c2*1/3,dtype=tf.float32)
        #c1,c2,c3,c4:[batch_size*3,4h,4w,1]

        #cal distance
        bmp_in=tf.transpose(bmp_in,(0,3,1,2))
        bmp_in=tf.reshape(bmp_in,[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2],1])
        #bmp_in:[batch_size*3,4h,4w,1]
        dis_c1=tf.square(bmp_in-c1)
        dis_c2=tf.square(bmp_in-c2)
        dis_c3=tf.square(bmp_in-c3)
        dis_c4=tf.square(bmp_in-c4)

        #get bmp_out
        dis=tf.concat([dis_c1,dis_c2,dis_c3,dis_c4],axis=3)
        index=tf.argmin(dis,axis=3,output_type=tf.int32)
        index=tf.reshape(index,[-1])
        temp=tf.range(tf.shape(index)[0])
        index=temp*4+index
        #index:[3batch_size*4h*4w]
    def _c1c2tobmp(self,c1,c2,bmp_in):
        c1_shape=tf.shape(c1)
        #put 3 channels to batch_size 
        c1=tf.transpose(c1,(0,3,1,2))
        c2=tf.transpose(c2,(0,3,1,2))
        c1=tf.reshape(c1,[c1_shape[0]*3,c1_shape[1],c1_shape[2]])
        c2=tf.reshape(c2,[c1_shape[0]*3,c1_shape[1],c1_shape[2]])
        #expand c1,c2 from batch_size*h*w to batch_size*4h*4w
        temp=tf.ones((1,1,1,4,4),dtype=tf.float32)
        c1=tf.expand_dims(c1,axis=3)
        c1=tf.expand_dims(c1,axis=4)
        c2=tf.expand_dims(c2,axis=3)
        c2=tf.expand_dims(c2,axis=4)
        c1=c1*temp
        c2=c2*temp
        c1=tf.reshape(tf.transpose(c1,(0,1,3,2,4)),[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2]])
        c2=tf.reshape(tf.transpose(c2,(0,1,3,2,4)),[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2]])
        c1=tf.expand_dims(c1,axis=3)
        c2=tf.expand_dims(c2,axis=3)
        #get c3,c4
        c3=tf.cast(c1*1/3+c2*2/3,dtype=tf.float32)
        c4=tf.cast(c1*2/3+c2*1/3,dtype=tf.float32)

        #cal distance
        bmp_in=tf.transpose(bmp_in,(0,3,1,2))
        bmp_in=tf.reshape(bmp_in,[c1_shape[0]*3,4*c1_shape[1],4*c1_shape[2],1])
        dis_c1=tf.square(bmp_in-c1)
        dis_c2=tf.square(bmp_in-c2)
        dis_c3=tf.square(bmp_in-c3)
        dis_c4=tf.square(bmp_in-c4)

        #get bmp_out
        dis=tf.concat([dis_c1,dis_c2,dis_c3,dis_c4],axis=3)
        index=tf.argmin(dis,axis=3,output_type=tf.int32)
        index=tf.reshape(index,[-1])
        c=tf.concat([c1,c2,c3,c4],axis=3)
        #c:[batch_size*3,4h,4w,4]
        c=tf.reshape(c,[-1])
        temp=tf.range(tf.shape(index)[0])
        index=temp*4+index
        bmp_out=tf.nn.embedding_lookup(c,index)
        bmp_out=tf.reshape(bmp_out,[c1_shape[0],3,c1_shape[1]*4,c1_shape[2]*4])
        # #put 3 channels back
        bmp_out=tf.transpose(bmp_out,(0,2,3,1))
        #bmp_out=bmp_in

        return bmp_out

    def _create_loss_without_padding(self):
        self.target_crop = utils.crop_center(self.target, tf.shape(self.bmp_prediction)[1:3])
        self.test2.append(self.target_crop)
        #bmp loss
        self.bmp_loss=tf.losses.mean_squared_error(self.target_crop,self.bmp_prediction)
        #c1c2 loss
        self.c1c2_loss=tf.losses.mean_squared_error(self.target_crop,self.c1c2_prediction)
        
        #regularization for c1c2
        max_bmp=tf.layers.max_pooling2d(self.bmp_prediction,4,4)
        min_bmp=tf.layers.max_pooling2d(-self.bmp_prediction,4,4)
        min_bmp=-min_bmp

        c1_loss=tf.square(min_bmp-self.c1)
        # self.test.append(min_bmp[0,10:11,10:11,1])
        # self.test.append(self.c1[0,10:11,10:11,1])
        # self.test.append(c1_loss[0,10:11,10:11,1])
        num=tf.cast(c1_loss>0,dtype=tf.float32)
        num=tf.reduce_sum(num)+10
        c1_loss=tf.reduce_sum(c1_loss)/num

        c2_loss=tf.square(self.c2-max_bmp)
        # self.test.append(max_bmp[0,10:11,10:11,1])
        # self.test.append(self.c2[0,10:11,10:11,1])
        # self.test.append(c2_loss[0,10:11,10:11,1])
        num=tf.cast(c2_loss>0,dtype=tf.float32)
        num=tf.reduce_sum(num)+10
        c2_loss=tf.reduce_sum(c2_loss)/num
        #self.loss=2*(c1_loss+c2_loss)

        #self.test.append(c1_loss)
        #self.test.append(c2_loss)
        #self.test.append(self.c1c2_loss)
        #self.test.append(self.bmp_loss)
        self.loss=self.bmp_loss+0.2*(self.c1c2_loss+10*(c1_loss+c2_loss))
        self.test.append(self.bmp_loss)
        self.test.append(self.c1c2_loss)
        self.test.append(c1_loss)
        self.test.append(c2_loss)
        #self.loss=self.bmp_loss
    def _create_optimizer(self):
        pass
        #you can put more optimizer here
        if(self.optimizer=='adam'):
            self.learning_rate=self.init_learning_rate*self.learning_rate_decay
            #self.test.append(self.learning_rate)
            optimizer=tf.train.AdamOptimizer(self.learning_rate)
        self.updates=optimizer.minimize(self.loss,self.global_step)

    def conv(self,x,hidden_size,bottleneck_size,name, with_padding = False):
        x=tf.nn.relu(x)
        x=tf.layers.conv2d(x,bottleneck_size,1,activation=tf.nn.relu,name=name+'_proj')
        if(with_padding):
            x = tf.layers.conv2d(x, hidden_size, 3, padding = 'same', activation=None,name=name+'_filt')
        else:
            x = tf.layers.conv2d(x, hidden_size, 3, activation=None, name=name+'_filt')
        return x

    

    def step(self,session,input,target,c1c2,learning_rate_decay,training=False):
        input_feed={}
        input_feed[self.input.name]=input
        input_feed[self.target.name]=target
        input_feed[self.c1c2_gt.name]=c1c2
        input_feed[self.learning_rate_decay.name]=learning_rate_decay
        if training:
            output_feed=[self.test1,self.test,self.test2,self.loss,self.updates]
        else:
            output_feed=[self.c1c2_prediction_cast,self.bmp_prediction_cast,self.test1]

        outputs=session.run(output_feed,input_feed)
        if(len(outputs)==1):
            outputs=[outputs[0],None]
        return outputs[0],outputs[1],outputs[2]
