import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
## zeno: import these addtional libraries to enable training with  mixed prevision 
from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import argparse
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(7) == labels[:,None]).astype(np.float32)
    return labels

class synthetic_data_gen:
    def __init__(self,num_samples  ,shape, num_class, batch_size):
        self.num_samples = num_samples
        self.shape = shape
        self.num_class=num_class
        self.batch_size=batch_size
    def numpy_tuples(self): 
        fake_img=np.random.randint(0, 255, size=self.shape).astype(set_numpy())             
        fake_label=np.array(random.randint(0,self.num_class-1)).astype(set_numpy()) 
        return fake_img, fake_label     

    def __call__(self):
        for sample_idx in range(self.num_samples):
            im,lb=self.numpy_tuples()
            lb = lb.reshape(1)
            im = im.reshape(shape)
            yield (im, lb) 

#print(x_.shape,y_.shape)
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):  
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix. 48x48
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix. 28x28
    conv2 = maxpool2d(conv2, k=2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4. 11x11
    conv3 = maxpool2d(conv3, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def adapt_types(im, lb):
    im = tf.cast(im, tf.float32)
    lb = tf.cast(lb, tf.uint8)
    return im, lb

def enforce_shape(im,lb):

    im=tf.reshape(im,shape=(-1,im_size,im_size*2,3))
    lb=tf.reshape(lb,shape=(-1,7))

    return im,lb

def one_hot(im,lb):
    lb = tf.one_hot(lb, num_class)
    return im, lb

def enforce_types(im, lb):
    im = tf.cast(im, dtype=get_dtype())
    lb = tf.cast(lb, dtype=get_dtype())
    return im, lb
    
def prep_synthetic_fun(total_num_records, shape, num_class, bz,use_aug,im_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE       
    dataset = tf.data.Dataset.from_generator(synthetic_data_gen(num_samples=total_num_records, shape=shape, num_class=num_class, batch_size=bz),
                                           output_types= (get_dtype(), get_dtype()), output_shapes= (
                           tf.TensorShape([im_size,im_size*2,3]), 
                           tf.TensorShape([1])))

    dataset = dataset.map(adapt_types)
    dataset = dataset.map(one_hot).batch(bz)
    dataset = dataset.map(enforce_shape)
    dataset = dataset.map(enforce_types)
    return dataset.repeat()

def get_dtype():
    #print("get_dtype fp16 used : ", args.use_fp16)
    return tf.float16 if args.use_fp16 else tf.float32
def set_numpy():
    if args.use_fp16:
        np_type=np.float16 
    else:
        np_type=np.float32
    return np_type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF1.15 synthetic data pipeline + model training with varying configurations')
    parser.add_argument('-bz', '--batch_size', default=8, type=int, metavar='N',
                        help='batch_size default 256')
    parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                        help='number of epoch to train the model default 50')
    parser.add_argument('--use_fp16', default=False, help='use FP16 for benchmarking', action='store_true')

    num2label = {0:'Anger', 1:'Disgust',2:'Fear', 3:'Happy',4: 'Neutral', 5:'Sad', 6:'Surprise'}
    label2num=dict([(b,a) for (a,b) in num2label.items()])
    args = parser.parse_args()
    bz=args.batch_size    
    global im_size
    im_size=1024
    shape=(im_size,im_size*2,3)
    num_channels=3
    num_class=len(num2label)
    epochs=int(args.epochs)
    total_num_records=1000 
    how_many_steps=int(total_num_records/bz)
    use_aug=False
    use_amp=args.use_fp16
    tf.compat.v1.disable_eager_execution()
    x_=np.random.randint(0, 255, size=(1,im_size,im_size*2,3)).astype(set_numpy()) 
    y_=np.array(random.randint(0,7-1)).reshape(1)
    y_=reformat(y_)
    y_=y_.astype(set_numpy())
    x = tf.compat.v1.placeholder(dtype=get_dtype(), shape=[None, im_size,im_size*2,3])
    y = tf.compat.v1.placeholder(dtype=get_dtype(), shape=[None, num_class])
    print(" ================= batch size : {} , synthetic data shape :{} , total records generated:{} , use amp :{} , epochs :{}, num_class:{} ================= ".format(str(bz),shape,str(how_many_steps), use_amp,str(epochs) , str(num_class)))
    learning_rate=0.001
    weights = {
    'wc1': tf.Variable(tf.random.truncated_normal([5,5,3,128],dtype=get_dtype())),
    'wc2': tf.Variable(tf.random.truncated_normal([5,5,128,64],dtype=get_dtype())),
    'wc3': tf.Variable(tf.random.truncated_normal([5,5,64,64],dtype=get_dtype())),
    'wd1': tf.Variable(tf.random.truncated_normal([128*256*64,64],dtype=get_dtype())),
    'out': tf.Variable(tf.random.truncated_normal([64,num_class],dtype=get_dtype())),
    }
    biases = {
    'bc1': tf.Variable(tf.random.truncated_normal([128],dtype=get_dtype())),
    'bc2': tf.Variable(tf.random.truncated_normal([64],dtype=get_dtype())),
    'bc3': tf.Variable(tf.random.truncated_normal([64],dtype=get_dtype())),
    'bd1': tf.Variable(tf.random.truncated_normal([64],dtype=get_dtype())),
    'out': tf.Variable(tf.random.truncated_normal([num_class],dtype=get_dtype())),
    }      
    logits = conv_net(x_, weights, biases)
    print("logits shape",logits.shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))    
    #print(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    #print(correct_prediction)
    #calculate accuracy across all the given images and average them out.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(accuracy)
    #optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.00001, momentum=0.9, use_nesterov=True).minimize(loss)
    optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(loss)
    ##### zeno: these are the addtional lines you need to add in order to enable mix precision
    if use_amp:
        tf.config.optimizer.set_experimental_options({'auto_mixed_precision':True})
        #config.set_optimizer_experimental_options({'auto_mixed_precision': True})
        mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled = True
    ##### zeno : adding one line to enable XLA
    tf.config.optimizer.set_jit(True) # Enable XLA
    # Initialize variables. local variables are needed to be initialized for tf.metrics.*
    init_g = tf.compat.v1.global_variables_initializer()
    init_l = tf.compat.v1.local_variables_initializer()  
    config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    ## make tf.dataset as data pipeline
    dataset=prep_synthetic_fun(total_num_records, shape, num_class, bz,use_aug,im_size)
    iterator = dataset.make_one_shot_iterator()
    training_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_training_element = training_iterator.get_next()
    training_init_op = training_iterator.make_initializer(dataset)
    start=time.time()
    with tf.device('/gpu:0'): 
        with tf.compat.v1.Session(config=config) as sess:
            sess.run([init_g, init_l])
            for epoch in range(epochs):  
                #print("steps should be ", how_many_steps)
                total_loss=0
                total_acc=0                
                for step in range(how_many_steps):
                    try:                    
                        sess.run(training_init_op) #initializing iterator here
                        bx, by = sess.run(next_training_element)
                        #print("batch shape",bx.shape, by.shape)
                        _, loss_, acc_ = sess.run([optimizer, loss, accuracy],feed_dict={x: bx, y: by})
                        #print("step: ",step, loss_,acc_)
                        total_loss+=loss_
                        total_acc+=acc_
                        #print('loss=%f acc=%f' % (loss_, acc_))
                    except tf.errors.OutOfRangeError:                    
                        pass
                print('epoch: %04d, loss=%f acc=%f' % (epoch, total_loss/(how_many_steps+1), total_acc/(how_many_steps+1)))

    end=time.time()
    duration=round(end-start,5)
    print("=================== training finished at {}===================".format(str(duration)))



