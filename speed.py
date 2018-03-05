import cv2
import numpy as np
import time
  
  #利用像素间差异性 运动物体速度求解:
  # 速度=相邻两帧的像素差/视频中的物体和实际物体的比例系数×每秒经过的数据帧数
  # V=PixelDiff/S*FPS           
cap=cv2.VideoCapture('filedir')
  #获得fps,总帧数
fps=cap.get(cv2.CAP_PROP_FPS)
print("fps",fps)
framenum=cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("framenum",framenum)
  #设置比例系数
s=0.2
  #计数器
num=0
flag=0
resize_width=512
resize_height=512
while(cap.isOpened()):  
      #遍历capture中的帧
      #指定视频属性
     
      cap.set(cv2.CAP_PROP_FRAME_WIDTH,resize_width)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT,resize_height)
     
      ret,frame=cap.read()
      #show the previous frame
      #将当前帧拷贝给img1把当前帧保存作为下一次处理的前一帧 
      if flag==0 :
           # 第一帧
             tempframe=frame
           #设置为灰度图
             tempgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
             flag=1
             num+=1
      else:
            # 计算当前帧和前一帧的不同,两幅图的差的绝对值输出到另一幅图上
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frameDelta=cv2.absdiff(frame,tempframe)
            # 阈值分割
            thresh=cv2.threshold(frameDelta,25,225,cv2.THRESH_BINARY)[-1]
            num+=1
