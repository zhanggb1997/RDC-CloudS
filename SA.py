# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/9/27 16:22 
@Author : 弓长广文武
======================================
"""
import torch
from torch import nn

'''
======================================
@File    :   SA.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
class ChannelAtten(nn.Module):
    def __init__(self, in_channels=2, ratio=16):
        super(ChannelAtten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//ratio,  in_channels, 1, bias=False)
        )
        self.sigm = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = self.shared_MLP(self.avg_pool(x))
        max = self.shared_MLP(self.max_pool(x))
        sig = self.sigm(avg + max)
        return x * sig
        # return out


class SpatialAtten(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=7, dila=2):
        super(SpatialAtten, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2*(dila), dilation=dila, bias=False)
        self.sigm = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max], dim=1)
        out = self.sigm(self.conv1(cat))
        # out = self.sigm(self.conv1(avg))
        return x * out
        # return out

class SpatialAtten1(nn.Module):
    def __init__(self, in_channels=2, mid_channels=1, kernel_size=3, dila=2):
        super(SpatialAtten1, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2 + dila // 2, dilation=dila, bias=True)
        self.conv3_ = nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2 + dila// 2, dilation=dila, bias=True)
        # self.conv1_ = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        self.sigm = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv3_ = self.conv3_(conv3)
        # conv1_ = self.conv1_(conv3)
        out = self.sigm(conv3_)
        return x * out
        # return out

class SpatialAtten2(nn.Module):
    def __init__(self, kernel_size=5, size=128):
        super(SpatialAtten2, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), bias=False)
        self.GAP_H = nn.AdaptiveAvgPool2d((size, 1))
        self.GMP_H = nn.AdaptiveMaxPool2d((size, 1))
        self.GAP_W = nn.AdaptiveAvgPool2d((1, size))
        self.GMP_W = nn.AdaptiveMaxPool2d((1, size))
        self.sigm = nn.Sigmoid()
        # self.AP_H = nn.AvgPool2d((size, 1))
        # self.AP_W = nn.AvgPool2d((1, size))
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # avg = torch.mean(x, dim=1, keepdim=True)
        # max, _ = torch.max(x, dim=1, keepdim=True)

        # cat = torch.cat([avg, max], dim=1)
        # out = self.sigm(self.conv1(cat))

        # add = avg + max
        # avg_h = torch.mean(add, dim=1, keepdim=True)
        # avg_w = torch.mean(add, dim=1, keepdim=True)

        avg_h = self.GAP_H(x)
        max_h = self.GMP_H(x)
        avg_w = self.GAP_W(x)
        max_w = self.GMP_W(x)

        # ap_h = self.AP_H(x)
        # ap_w = self.AP_W(x)

        # avg_h = torch.mean(x, dim=-2, keepdim=True)
        # avg_w = torch.mean(x, dim=-1, keepdim=True)

        h_avg = torch.mean(avg_h, dim=1, keepdim=True)
        h_max, _ = torch.max(max_h, dim=1, keepdim=True)
        h = (h_avg + h_max)
        # h = torch.cat((h_avg, h_max),  dim=1)

        w_avg = torch.mean(avg_w, dim=1, keepdim=True)
        w_max, _ = torch.max(max_w, dim=1, keepdim=True)
        w = (w_avg + w_max)
        # w = torch.cat((w_avg, w_max), dim=1)

        # max, _ = torch.max(w, dim=1, keepdim=True)

        h_conv = self.conv(h.squeeze(-1)).unsqueeze(-1)
        w_conv = self.conv(w.squeeze(-2)).unsqueeze(-2)

        # h_conv = self.sigm(h_conv)
        # w_conv = self.sigm(w_conv)
        #
        h_w = h_conv * w_conv
        h_w = self.sigm(h_w)

        return h_w * x
        # return x * out
        # return out


class SpatialAtten3(nn.Module):
    def __init__(self, kernel_size=5, size=128):
        super(SpatialAtten3, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=((kernel_size - 1) // 2), bias=False)
        self.GAP_H = nn.AdaptiveAvgPool2d((size, 1))
        self.GAP_W = nn.AdaptiveAvgPool2d((1, size))
        self.sigm = nn.Sigmoid()
        # self.AP_H = nn.AvgPool2d((size, 1))
        # self.AP_W = nn.AvgPool2d((1, size))
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # avg = torch.mean(x, dim=1, keepdim=True)
        # max, _ = torch.max(x, dim=1, keepdim=True)

        # cat = torch.cat([avg, max], dim=1)
        # out = self.sigm(self.conv1(cat))

        # add = avg + max
        # avg_h = torch.mean(add, dim=1, keepdim=True)
        # avg_w = torch.mean(add, dim=1, keepdim=True)

        avg_h = self.GAP_H(x)
        avg_w = self.GAP_W(x)

        # ap_h = self.AP_H(x)
        # ap_w = self.AP_W(x)

        # avg_h = torch.mean(x, dim=-2, keepdim=True)
        # avg_w = torch.mean(x, dim=-1, keepdim=True)

        h = torch.mean(avg_h, dim=1, keepdim=True)
        w = torch.mean(avg_w, dim=1, keepdim=True)
        # max, _ = torch.max(w, dim=1, keepdim=True)

        h_conv = self.conv(h.squeeze(-1)).unsqueeze(-1)
        w_conv = self.conv(w.squeeze(-2)).unsqueeze(-2)

        h_w = self.sigm(h_conv * w_conv)
        # w_sig = self.sigm(w_conv)

        return h_w * x
        # return x * out
        # return out