# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:24:29 2018

@author: yanghang
"""
#%%
import tensorflow as tf
import numpy as np
import os

#%%
# step 1.获取所有的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir):
    '''
    Args:
        file_dir:file directory
    Return:
        list of imgs and labels
   index- image_list: value=[image]
        - label_list: value=[label]
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

train_dir='C:\\Users\\yanghang\\ugthesis\\data'
image_list,label_list=get_file(train_dir)

# step 2 将生成的List传入到batch中，因为img和label是分开的所以用tf.train.slice_input_producer([image,label])
# 但是step 2的代码现在调试还有问题
def get_batch(image,label,new_height,new_width,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(image,tf.int32)

    input_queue= tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)

   image=tf.image.resize_images(image,(new_height,new_width))
   image=tf.image.per_image_standardization(image)
   image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,capacity=capacity,num_threads=8)

   label_batch=tf.reshape(label_batch,[batch_size])
   image_batch=tf.cast(image_batch,tf.float32)
return image_batch,label_batch

#或者根据stackflow上面的建议
#https://stackoverflow.com/questions/48729620/python-tensorflowunimplementederror-cast-string-to-int32-is-not-supported?noredirect=1#comment84475634_48729620，
#用tf.dataset来取代把用队列
filenames=["train.tfrecords"]
dataset=tf.data.TFRecordDataset(filenames)
def _parse_function(record):
    features={"image":tf.FixedLenFeature((),tf.string,default_value='')
              "label":tf.FixedLenFeature((),tf.int32,default_value='')}
# use tf.parse_single_example() function to extract the data            
    parsed=tf.parse_single_example(record,features)
    image_decoded=tf.image.decoded_image(parsed["image"]) 
    image_resize=tf.image.reshape(image_decoded,[200,200,1])
    label=tf.cast(parse["label"],tf.int32)
 return {"image":image,"label":label}
 dataset=dataset.map(_parse_function) #parse the record into tensor
 dataset=dataset.shuffle(buffer_size=1000)
 dataset=dataset.batch(32)
 iterator=dataset.make_initializable_iterator()
 image,label=iterator.get_next()

#compute for 10 epochs
 for _in range(100):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break

  





