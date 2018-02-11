# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:05:36 2018

@author: yanghang
"""

#%%

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:05:36 2018

@author: yanghang
"""

#%%

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
'''
读取图片
subtract mean from image
tf.image.per_image_standardization()
Function:
     computes (x - mean) / adjusted_stddev, 
     where mean is the average of all values in image, 
     adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements())).
'''


def get_files(file_dir):
    '''
    Return the list of image and label
    '''
    lahu=[]
    label_lahu=[]
    normal=[]
    label_normal=[]
    for item in os.listdir(file_dir):
        name=item.split(sep='.')
        if name[0]=='lahu':
           lahu.append(file_dir+item)
           label_lahu.append(1)
        else:
           normal.append(file_dir+item)
           label_normal.append(2)
    print('There are %d lahu|n There are %d normal' %(len(lahu),len(normal)))
    image_list=np.hstack((lahu,normal))
    label_list=np.hstack((label_lahu,label_normal))
    temp=np.array([image_list,label_list])
    temp=temp.transpose()
    np.random.shuffle(temp)
    
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[int(i) for i in label_list]
    return image_list,label_list


def get_batch(image,label,new_height,new_width,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(image,tf.float32)
    # tf.train.slice_input_producer
    # Args: tensor_list: a list of tensor_list
    input_queue= tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)
    '''
    实现对图片的预处理比如 tf.image.rgb_to_grayscale, tf.image.random_flip_left_right
    '''
    # the data we want to batch must have pre-defined shape or we need to specify the shape 
    # otherwise they will be TensorShape(Dimenson(None))
    image=tf.image.resize_images(image,(new_height,new_width))
    image=tf.image.per_image_standardization(image)
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,capacity=capacity,num_threads=8)
    
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

BATCH_SIZE=2
CAPCITY=256
new_height=200
new_width=200
train_dir='C:\\Users\\yanghang\\ugthesis\\data\\b'
image_list,label_list=get_files(train_dir)
image_batch,label_batch=get_batch(image_list,label_list,new_height,new_width,BATCH_SIZE,CAPCITY)


with tf.Session() as sess:
    i=0
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)#启动线程
    try:
        while not coord.should_stop() and i<2:
            image,label=sess.run([image_batch,label_batch])
            for j in np.arrange(BATCH_SIZE):
                plt.imshow(image[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
         print('Done')
    #print(sess.run(image1))
    finally:
         coord.request_stop()
    
coord.join(threads)
sess.close()
