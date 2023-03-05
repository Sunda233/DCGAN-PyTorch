
import argparse
import os
import numpy as np
import math
import torch
from click.core import F
from torch.nn.modules.loss import _Loss
from torch import nn, Tensor
import math
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader


true = 1
pred = 22



# 均方误差（Mean Square Error，MSE）
def mse(true, pred):
    return np.sum((true - pred) ** 2)

# 平均绝对误差（ Mean Absolute Error,MAE)
def mae(true, pred):
    return np.sum(np.abs(true - pred))  # 绝对值

def sdl(true,pred):
    return math.sqrt(((true - pred) ** 2)/1)


print(mse(true, pred))
print(mae(true, pred))
print(sdl(true, pred))

# 铰链损失（Hinge Loss）
def update_weights_Hinge(m1, m2, b, X1, X2, Y, learning_rate):
    m1_deriv = 0
    m2_deriv = 0
    b_deriv = 0
    N = len(X1)
    for i in range(N):
     # Calculate partial derivatives
        if Y[i] * (m1 * X1[i] + m2 * X2[i] + b) <= 1:
            m1_deriv += -X1[i] * Y[i]
            m2_deriv += -X2[i] * Y[i]
            b_deriv += -Y[i]
    # else derivatives are zero
    # We subtract because the derivatives point in direction of steepest ascent
            m1 -= (m1_deriv / float(N)) * learning_rate
            m2 -= (m2_deriv / float(N)) * learning_rate
            b -= (b_deriv / float(N)) * learning_rate
            print("铰链损失")
            print(m1, m2, b)
    return m1, m2, b


# importing requirements

# from keras.layers import Dense
# from keras.models import Sequential
# from keras.optimizers import adam
# # alpha = 0.001 as given in the lr parameter in adam() optimizer
#
# # build the model
#
# model_alpha1 = Sequential()
# model_alpha1.add(Dense(50, input_dim=2, activation='relu'))
# model_alpha1.add(Dense(3, activation='softmax'))
# # compile the model
# opt_alpha1 = adam(lr=0.001)
# model_alpha1.compile(loss='categorical_crossentropy', optimizer=opt_alpha1, metrics=['accuracy'])
# # fit the model
# # dummy_Y is the one-hot encoded
# # history_alpha1 is used to score the validation and accuracy scores for plotting
# history_alpha1 = model_alpha1.fit(dataX, dummy_Y, validation_data=(dataX, dummy_Y), epochs=200, verbose=0)


class SmoothL1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.beta = beta
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


import torch
import torch.nn as nn
import math


# SmoothL1Loss 例子

def validate_loss(output, target, beta):
    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            if math.fabs(x - y) < beta:
                loss_val = 0.5 * math.pow(x - y, 2) / beta
            else:
                loss_val = math.fabs(x - y) - 0.5 * beta
            val += loss_val
    return val / output.nelement()

beta = 1
loss_fct = nn.SmoothL1Loss(reduction="mean", beta=beta)
input_src = torch.Tensor([[0.8, 0.8], [0.9, 0.9], [0.3, 0.3]])
target = torch.Tensor([[0.6, 0.6], [0.7, 0.8], [0.4, 0.5]])
print(input_src.size())
print(target.size())
loss = loss_fct(input_src, target)
print(loss.item())

validate = validate_loss(input_src, target, beta)
print(validate)

loss_fct = nn.SmoothL1Loss(reduction="none", beta=beta)
loss = loss_fct(input_src, target)
print(loss)


# IoU Los 实例
# def Iou(box1, box2, wh=False):
#     if wh == False:
# 	    xmin1, ymin1, xmax1, ymax1 = box1
# 	    xmin2, ymin2, xmax2, ymax2 = box2
#     else:
# 	    xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
# 	    xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
# 	    xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
# 	    xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
#     # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
#     xx1 = np.max([xmin1, xmin2])
#     xx2 = np.min([xmax1, xmax2])
#     yy1 = np.max([ymin1, ymin2])
#     yy2 = np.min([ymax1, ymax2])
#     # 计算两个矩形框面积
#     area1 = (xmax1-xmin1) * (ymax1-ymin1)
#     area2 = (xmax2-xmin2) * (ymax2-ymin2)
#     inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))　#计算交集面积
#     iou = inter_area / (area1+area2-inter_area+1e-6) 　#计算交并比
#     return iou


# def Giou(rec1,rec2):
#     #分别是第一个矩形左右上下的坐标
#     x1,x2,y1,y2 = rec1
#     x3,x4,y3,y4 = rec2
#     iou = Iou(rec1,rec2)
#     area_C = (max(x1,x2,x3,x4)-min(x1,x2,x3,x4))*(max(y1,y2,y3,y4)-min(y1,y2,y3,y4))
#     area_1 = (x2-x1)*(y1-y2)
#     area_2 = (x4-x3)*(y3-y4)
#     sum_area = area_1 + area_2
#
#     w1 = x2 - x1   #第一个矩形的宽
#     w2 = x4 - x3   #第二个矩形的宽
#     h1 = y1 - y2
#     h2 = y3 - y4
#     W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)    #交叉部分的宽
#     H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)    #交叉部分的高
#     Area = W*H    #交叉的面积
#     add_area = sum_area - Area    #两矩形并集的面积
#
#     end_area = (area_C - add_area)/area_C    #闭包区域中不属于两个框的区域占闭包区域的比重
#     giou = iou - end_area
#     return giou



