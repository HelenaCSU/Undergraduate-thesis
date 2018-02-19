# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:05:42 2018

@author: yanghang
"""
import tensorflow as tf
import numpy as np
#%%
def conv(layer_name,x,out_channels,kernel_size=[3,3],stride=[1,1,1,1],is_pretrain=True):
    '''
    Conv ops
    Args:
        Layer_name: e.g. conv1,pool1
        x:input
    Return:
        4D tensor
    '''
    in_channels=x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        # Args of tf.get_variable(name,shape,dtype
        #                         initializer,regularizer,trainable,collections
        #                         caching_device)
        w=tf.get_variable(name='weights',trainable=is_pretrain,
                          shape=[kernel_size[0],kernel_size[1],in_channels,out_channels],
                          initializer=tf.contrib.layers.xavier_initializer)
        b=tf.get_variable(name='biases',trainable=is_pretrain,shape=[out_channels],
                          intializer=tf.constant_initializer(0.0))
        # computes a 2-D convolution given 4-D input and filter tensor
       # Args of x=tf.nn.conv2d(input,filter,stride,padding)
       #input=[batch,height,width,channel]; filter=[filter_height,filter_width,input_channel,output_channel]
        x=tf.nn.conv2d(x,w,stride,padding='SAME',name='conv')
        x=tf.nn.bias_add(x,b,name='bias_add')
        x=tf.nn.relu(x,name='relu')
        return x
#%%
def pool(layer_name,x,kernel_size=[1,2,2,1],stride=[1,2,2,1],is_max_pool=True):
    '''
    Pool ops
    '''
    #tf.nn.max_pool(value,ksize,stride,padding,name)
    if is_max_pool:
        x=tf.nn.max_pool(x,kernel_size,stride,padding='SAME',name=layer_name)
    else:
        x=tf.nn.avg_pool(x,kernel_size,stride,padding='SAME',name=layer_name)
    return x
#%%
def batch_norm(x):
    '''
    normalization ops
    '''  
    variance_epsilon=1e-3
    batch_mean,batch_variance=tf.nn.moments(x,[0])
    x=tf.nn.batch_normalization(x,batch_mean,batch_variance,offset=None,scale=None,variance_epsilon)
    return x
#%%
def FC_layers(layer_name,out_nodes):
    shape=x.get_shape()
    if len(shape)==4:
        size=shape[1].value*shape[2].value*shape[3].value
    else:
        size=shape[-1].value
        
   