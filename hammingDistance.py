# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from PIL import Image


# calculate the delete the same image
def difference(imagename):
    '''
    param:PIL imagename
    return 差值array, 0&1
    '''
    resize_width=9
    resize_height=8
    image=Image.open(imagename)
    image=image.convert('L').resize((resize_width,resize_height),Image.ANTIALIAS)
    pixels=list(image.getdata())
    difference=[]
    for row in range(resize_height):
        row_start_index=row*(resize_width)
        for col in range(resize_width-1):
            pixel_left_index=row_start_index+col
            difference.append(pixels[pixel_left_index]>pixels[pixel_left_index+1])
    return difference

def hash_value(imagename):
    '''
    param: PIL.image
    return: dhash(str)
    '''
    #convert the binary array to hex string
    difference1 = difference(imagename)
    decimal_value=0
    hex_string=""
    for index,value in enumerate(difference1):
        if value:
    #str.rjust(width[,fillchar])
    #width total string length
           decimal_value+=value*(2**(index% 8))
           if(index%8)==7:
               hex_string+=str(hex(decimal_value)[2:].rjust(2,"0"))
               decimal_value=0
    return hex_string


def hammingDistance(dhash1,dhash2):
    """
    param dhash1:str
    param dhash2:str
    return hamming Dist
    """
    #汉明距离示两个（相同长度）字对应位不同的数量 
    #对两个字符串进行异或运算，并统计结果为1的个数，那么这个数就是汉明距离   

    difference=(int(dhash1,16))^(int(dhash2,16))
    return bin(difference.count("1"))

def getImage():
    FindPath="./"
    filenames=os.listdir(FindPath)
    for name in filenames:
        filePath=os.path.join(FindPath,name)
        print(filePath)
  

if __name__ == "__main__":
   getImage()

    

     

