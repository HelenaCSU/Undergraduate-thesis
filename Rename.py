# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:48:02 2018

@author: yanghang
"""
import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    用try,except语句来捕捉出现的所有故障
    '''
    def __init__(self):
        self.path = 'C:\\Users\\yanghang\\ugthesis\\d'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 342
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'normal_'+str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()