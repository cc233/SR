import tensorflow as tf
from utils import utils

class naive_SR(object):
    def __init__(self,
                 hidden_size,
                 bottleneck_size,
                 learning_rate,
                 optimizer='adam',
                 dtype=tf.float32,
                 scope='naive_SR',
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
            self.build_graph()
            self.saver = tf.train.Saver(max_to_keep=3)
        #for var in tf.global_variables():
        #    print var
    def build_graph(self):
        self._create_placeholder()
        if self.with_padding:
            self._create_struct()
            self._create_loss()
        else:
            self._create_struct_without_padding()
            self._create_loss_without_padding()
        self._create_optimizer()
    
    def _create_placeholder(self):
        pass
        self.lr_input=tf.placeholder(self.dtype,[None,None,None,3],name='lr_input')
        self.target=tf.placeholder(self.dtype,[None,None,None,3],name='output')
        self.learning_rate_decay=tf.placeholder(tf.float32,name='learning_rate_decay')
    def _create_loss1(self):
        pass
        x=tf.layers.conv2d(self.lr_input,self.hidden_size,1,activation=None,name='in')
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
        self.prediction=tf.layers.conv2d(x,3,1,name='out')
        self.target_crop=utils.crop_center(self.target,tf.shape(self.prediction)[1:3])
        #self.test=tf.reduce_mean(self.prediction)
        #self.test=tf.reduce_mean(self.prediction)
        self.loss = tf.losses.mean_squared_error(self.target_crop, self.prediction)
        #self.loss = tf.losses.log_loss(self.target_crop, self.prediction)
        #self.loss = tf.losses.absolute_difference(self.target_crop, self.prediction)

    def _create_struct(self):
        pass
        x=tf.layers.conv2d(self.lr_input,self.hidden_size,1,activation=None,name='bmp_in')
        #self.test=tf.reduce_mean(x)
        #low resolution
        for i in range(6):
            x=x+self.conv(x,self.hidden_size,self.bottleneck_size,'lr_conv'+str(i), with_padding=True)
        temp=tf.nn.relu(x)
        #up sampling
        x=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2)+tf.layers.conv2d_transpose(temp,self.hidden_size,2,strides=2,name='up_sampling')

        #high resolution
        for i in range(4):
            x=x+self.conv(x,self.hidden_size,self.bottleneck_size,'hr_conv'+str(i), with_padding=True)
        x=tf.nn.relu(x)
        # self.prediction = tf.layers.conv2d(x, 3, 1, padding = 'same', name='out')
        # still not sure about here
        self.bmp_prediction=tf.layers.conv2d(x,3,1,name='out')

    def _create_struct_without_padding(self):
        x=tf.layers.conv2d(self.lr_input,self.hidden_size,1,activation=None,name='in')
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
        bicubic=tf.image.resize_bicubic(self.lr_input+128, tf.shape(self.lr_input)[1:3] * 2, name='bicubic')
        bicubic=utils.crop_center(bicubic,tf.shape(x)[1:3])
        self.bmp_prediction=x+bicubic
        self.bmp_prediction_cast=tf.saturate_cast(self.bmp_prediction, tf.uint8)
        #self.target_crop=utils.crop_center(self.target,tf.shape(self.bmp_prediction)[1:3])

    def test_func(self):
        a1=tf.range(9)
        a1=tf.reshape(a1,[3,3,1])
        a2=a1+9
        a3=a1+18
        a=tf.concat([a1,a2,a3],axis=2)
        b=tf.range(9)
        one=tf.ones([1])
        b=b%3
        d=tf.range(9)
        a=tf.reshape(a,[9,3])
        c=a[0,b]
        c=tf.reshape(c,[3,3])
        self.test.append(c)

    def _create_loss_real_resnet(self):
        pass

        x = tf.layers.conv2d(self.lr_input, self.hidden_size, 1, activation=None, name='in')
        # self.test=tf.reduce_mean(x)
        # low resolution
        for i in range(6):
            x = utils.crop_by_pixel(x, 1) + self.conv(x, self.hidden_size, self.bottleneck_size, 'lr_conv' + str(i))
        # temp = tf.nn.relu(x)
        # up sampling
        x = tf.image.resize_nearest_neighbor(x, tf.shape(x)[1:3] * 2) + tf.layers.conv2d_transpose(x,
                                                                                                   self.hidden_size, 2,
                                                                                                   strides=2,
                                                                                                   name='up_sampling')

        # high resolution
        for i in range(4):
            x = utils.crop_by_pixel(x, 1) + self.conv(x, self.hidden_size, self.bottleneck_size, 'hr_conv' + str(i))

        x = tf.layers.conv2d(x, 3, 1, name='out')
        # bicubic upsampling
        input_cropped = utils.crop_center(self.lr_input, tf.cast(tf.shape(x)[1:3] / self.scale, dtype = tf.int32))
        bicubic = tf.image.resize_bicubic(input_cropped, tf.shape(input_cropped)[1:3] * 2, name='bicubic')
        self.prediction = x + bicubic
        # self.test=tf.reduce_mean(self.prediction)
        # self.test=tf.reduce_mean(self.prediction)
        self.target_crop = utils.crop_center(self.target, tf.shape(self.prediction)[1:3])
        self.loss = tf.losses.mean_squared_error(self.target_crop, self.prediction)
        # self.loss = tf.losses.log_loss(self.target_crop, self.prediction)
        # self.loss = tf.losses.absolute_difference(self.target_crop, self.prediction)

    def _create_loss(self):
        #bmp loss
        self.bmp_loss=tf.losses.mean_squared_error(self.target,self.bmp_prediction)
        self.loss=self.bmp_loss
    def _create_loss_without_padding(self):
        self.target_crop = utils.crop_center(self.target, tf.shape(self.bmp_prediction)[1:3])
        self.loss = tf.losses.mean_squared_error(self.target_crop, self.bmp_prediction)
    def _create_optimizer(self):
        pass
        #you can put more optimizer here
        if(self.optimizer=='adam'):
            self.learning_rate=self.init_learning_rate*self.learning_rate_decay
            self.test.append(self.learning_rate)
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

    

    def step(self,session,lr_input,target,learning_rate_decay,training=False):
        input_feed={}
        input_feed[self.lr_input.name]=lr_input
        input_feed[self.target.name]=target
        input_feed[self.learning_rate_decay.name]=learning_rate_decay
        if training:
            output_feed=[self.test,self.loss,self.updates]
        else:
            output_feed=[self.bmp_prediction_cast]

        outputs=session.run(output_feed,input_feed)
        if(len(outputs)==1):
            outputs=[outputs[0],None]
        return outputs[0],outputs[1]
