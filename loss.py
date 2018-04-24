import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import re
import math
from mpl_toolkits.axes_grid1 import host_subplot
import  pylab
from pylab import figure,show,legend

fp=open('faster_rcnn_alt_opt_ZF_.txt','r')

train_iterations=[]
train_loss=[]
bbox_loss=[]
cls_loss=[]
rpn_cls_loss=[]
rpn_loss_bbox=[]


if __name__ == '__main__':

        with open('faster_rcnn_alt_opt_ZF_.txt', 'r') as f:
            lns = fp.readlines()

        for ln in lns:
            ln = ln.strip()
            train_iterations_value = re.findall(r'Iteration (\d+?), loss = (\d.\d+)', ln)
            bbox_loss_value = re.findall(r'#0: bbox_loss = (\d.\d+)', ln)
            cls_loss_value = re.findall(r'#1: cls_loss = (\d.\d+)', ln)
            # rpn_cls_loss_value=re.findall(r' #0: rpn_cls_loss= (\d.\d+)',ln)
            # rpn_loss_bbox_value=re.findall(r' # 1: rpn_loss_bbox=(\d.\d+)',ln)


            if train_iterations_value:
                iteration = list(train_iterations_value[0])[0]
                loss = list(train_iterations_value[0])[1]
                train_iterations.append(int(iteration))
                train_loss.append(float(loss))

            if bbox_loss_value:
                bbox_loss.append(bbox_loss_value[0])
                # print(bbox_loss)
            if cls_loss_value:
                cls_loss.append(float(cls_loss_value[0]))


        first=train_iterations[0:40]
        first_loss=train_loss[0:40]
        second=train_iterations[40:60]
        second_loss=train_loss[40:60]
        third=train_iterations[60:100]
        third_loss=train_loss[60:100]
        fourth=train_iterations[100:120]
        fourth_loss=train_loss[100:120]


        host = host_subplot(121)
        # plt.subplots_adjust(right=0.8)
        host.set_xlabel("iterations")
        host.set_ylabel("loss")
        p1, = host.plot(first, first_loss, label='loss1')
        p2, = host.plot(second,second_loss,label='loss2')
        p3, = host.plot(third,third_loss,label='loss3')
        p4, = host.plot(fourth,fourth_loss,label='loss4')
        host.legend(loc=1)
        host.axis["left"].label.set_color(p1.get_color())
        host.set_xlim([0, 1000])
        host.set_ylim([0., 1.5])

        host2=host_subplot(122)
        # plt.subplots_adjust(left=0.2)
        host2.set_xlabel("iterations")
        host2.set_ylabel("cls_loss")
        # p5, = host2.plot(first,bbox_loss,label='bbox_loss')
        p6, = host2.plot(first,cls_loss,label='cls_loss')
        host.legend(loc=1)
        host.axis["right"].label.set_color(p2.get_color())
        host2.set_xlim([0,1000])
        host2.set_ylim([0,1.5])
        plt.draw()
        plt.show()
