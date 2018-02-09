# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:05:36 2018

@author: yanghang
"""

#%%

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

'''
读取图片
subtract mean from image
tf.image.per_image_standardization()
Function:
     computes (x - mean) / adjusted_stddev, 
     where mean is the average of all values in image, 
     adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements())).
'''
path='C:\\Users\\yanghang\\ugthesis\\data\\a\\lahu38.jpg'
file_queue=tf.train.string_input_producer([path])
image_reader=tf.WholeFileReader()
_,image=image_reader.read(file_queue)
image=tf.image.decode_jpeg(image)

with tf.Session() as sess:
    tf.global_variables_initializer()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)#启动线程
   
    image1=tf.image.per_image_standardization(image)
    plt.imshow(image1)
    #print(sess.run(image1))
    coord.request_stop()
    coord.join(threads)
