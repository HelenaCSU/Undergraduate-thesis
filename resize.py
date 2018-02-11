# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:55:16 2018

@author: yanghang
"""
#%%
from PIL import Image # PIL Python Image Library
import os
import sys
#%%
'''
对图片进行缩放(既可以是缩小到指定的width，height； 也可以缩小到Times(倍数)

os.listdir() 是返回指定目录下的所有文件和目录名
os.path.join(path,name)是连接目录与文件名
os.path.splitext() 是分离文件名和扩展名
'''
path ="C:\\Users\\yanghang\\ugthesis\\data\\a\\" # 要在结尾加\\
dirs=os.listdir(path)

def resize():
    for item in dirs:
       if item.split('.')[-1]=='jpg':
            item_path=os.path.join(path,item)
            im=Image.open(item_path)
            #(width,height)=im.size
            #imResize1=im.resize((width*Times,height*Times),Image.ANTIALIAS)
            f,e=os.path.splitext(item_path)
            imResize=im.resize((224,224),Image.ANTIALIAS)
            imResize.save(f+'resized.jpg','JPEG', quality=90)

resize()
# 在TensorFlow里面可以利用tf.image.resize_image_with_crop_or_pad() 代替上面一段代码
# Args: image(4-D tensor[batch,height,width,channel]) ; target_height;target_width
