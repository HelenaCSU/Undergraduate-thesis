# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:24:29 2018

@author: yanghang
"""
#%%
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io

#%%

def get_files(file_dir):
    '''
    Args:
        file_dir:file directory
    Return:
        list of imgs and labels
    '''
    images=[]
    temp=[]
   
    for root,sub_folders,files in os.walk(file_dir):
        #imge directories
        for name in files:
            images.append(os.path.join(root,name)) #.append(obj)将obj添加到list后面， os.path.join(a,b)起连接作用
        #sub-folder names:
        for name in sub_folders:
            temp.append(os.path.join(root,name))
         
    labels=[]
    for one_folder in temp:
        n_img=len(os.listdir(one_folder))
        letter=one_folder.split('\\')[-1]
        
        if(letter=='a'):
            labels=np.append(labels,n_img*[1])
        else:
            labels=np.append(labels,n_img*[2])
    
    #shuffle
    temp=np.array([images,labels])
    temp=temp.transpose()
    np.random.shuffle(temp)
    
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[int(float(i)) for i in label_list]
    
    return image_list,label_list
'''
the format of img: byteslist
the format of label: int64
'''
def int64_feature(value):
 
    return tf.train.Feature(int64_list=tf.train.Int64list(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(byte_list=tf.train.BytesList(value=[value]))
#%%

def convertToTFRecords(images,labels,save_dir,name):
    '''
    Args:
        images: list of image
        labels: list of label
        name : the name of tfrecord file,string type
        用tf.train.Example来定义我们要填入的数据格式
        用tf.python_io.TFRecordWriter来写入
    '''
    
    filename=os.path.join(save_dir,name+',tfrecords') # os.path.join来拼接路径，str+str 
    n_samples=len(labels)
    
    if np.shape(images)[0] !=n_samples:
        raise ValueError('Images size %d does not match label size %d')
        
    writer=tf.python_io.TFRecordWriter(filename)
    print('\nTransform start...')
    for i in np.arange(0,n_samples):
        try:
            image=io.imread(images[i])
            image_raw=image.tostring() #把数据转为原生的bytes
            label=int(labels[i])
            example=tf.train.Example(features=tf.train.Features(features={'label':int64_feature(label),'images':bytes_feature(image_raw)}))
            writer.write(example.SerializetoString())
        except:
            continue
    writer.close()
    print('transform done')

#%%   

def read_and_decode(tfrecords_file,batch_size):
    '''
    Args:
        生成的TFRecord文件
        Batch的大小
    Return:
        a dict mapping feature keys to tensor
    '''
    '''    
    tf.train.string_input_producer(string_tensor,    #1-D string tensor
    num_epochs=None,  #迭代次数
    shuffle=True,     
    seed=None,
    capacity=32,      # queue capacity
    shared_name=None,
    name=None,
    cancel_op=None
    
    tf.parse_single_example(serialized,
    features, # 含有FixedLenFeature特征值的字典
    name=None,
    example_name=None)
)
    '''
    filename_queue=tf.train.string_input_producer([tfrecords_file])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    #tf.parse_single_example是解析器，把example的内存解析为张量
    img_features=tf.parse_single_example(serialized_example,features={'label':tf.FixedLenFeature([],tf.int64),'image_raw':tf.FixedLenFeature([],tf.string)})
    image=tf.decode_raw(img_features['image_raw'],tf.uint8)
    
    image=tf.reshape(image,[224,224])
    label=tf.cast(img_features['label'],tf.int32)
    '''
    tensor: the list or dictionary of tensor tp enqueue
    capcaity:the max number of elems in the queue
    num_threads: the number of thread in the queue
    '''
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=100)
    return image_batch, tf.reshape(label_batch,[batch_size])

#%% convert data to TFRecords

test_dir='C:\\Users\\yanghang\\ugthesis\\data'
save_dir='C:\\Users\\yanghang\\ugthesis'
BATCH_SIZE=25
name_test='test'
images,labels=get_files(test_dir)
convertToTFRecords(images,labels,save_dir,name_test)
tfrecords_file='test,tfrecords'
read_and_decode(tfrecords_file,BATCH_SIZE)