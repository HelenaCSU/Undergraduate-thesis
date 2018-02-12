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
