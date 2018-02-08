# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:29:48 2018

@author: yanghang
"""
import cv2
'''
从pathIn的路径里读取视频
分解的图片存储在pathOut里面

'''
def extractImages(pathIn,pathOut):
   
    
   vidcap=cv2.VideoCapture(pathIn)

   timeF=1 # 
   success,image=vidcap.read()
   count=0 
   i=1
   success=True
   while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))
    success,image=vidcap.read()
    print('Read a new frame',success)
    if(count%timeF==0):
       cv2.imwrite("img%d.jpg" % i,image)
       i+=1
    count+=1
    
  
pathIn='lahu1.avi';
pathOut='folder path'
extractImages(pathIn,pathOut)
