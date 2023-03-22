# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/9/19 9:26 
@Author : 弓长广文武
======================================
"""
import torch
from torch import nn

'''
======================================
@File    :   CA.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
class ECA_Layer(nn.Module):
    def __init__(self, kernel_size):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # y_max = self.max_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        # return y.expand_as(x)

class ECA_Layer1(nn.Module):
    def __init__(self, kernel_size):
        super(ECA_Layer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        # Two different branches of ECA module
        # y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        # y_avg = self.sigmoid(y_avg)

        y_max = self.max_pool(x)
        # Two different branches of ECA module
        # y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        # y_max = self.sigmoid(y_max)

        # y = y_avg + y_max
        y = torch.cat([y_avg, y_max], dim=2)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # y = self.sigmoid(y_max + y_avg)
        y = self.sigmoid(y)
        # y = y_max + y_avg

        return x * y.expand_as(x)
        # return y.expand_as(x)
