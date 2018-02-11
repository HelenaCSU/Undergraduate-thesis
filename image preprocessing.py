# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:06:01 2018

@author: yanghang
"""

import cv2
import numpy
from matplotlib import pyplot as plt
path='C:\\Users\\yanghang\\ugthesis\\data\\a\\lahu38.jpg'
image=cv2.imread(path)
blur=cv2.GaussianBlur(image,(3,3),0)
while(1):
    cv2.imshow('image',image)
    cv2.imshow('blur',blur)
    k=cv2.waitKey(1)
    if k== ord('q'):
        break
cv2.destroyAllWindows()
print(image.shape)
q