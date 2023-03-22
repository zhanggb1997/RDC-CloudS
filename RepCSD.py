# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/9/13 19:38 
@Author : 弓长广文武
======================================
"""
from torch import nn

from Rep_code.SA import SpatialAtten
from Rep_code.CA import ECA_Layer
from Rep_code.RepCSD_utils import MFF_Block, OUT_Block, DS_Block

'''
======================================
@File    :   RepCSD.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
class RepCSD(nn.Module):
    def __init__(self, num_classes=2, num_channels_list=None, use_atten=False, deploy=False, k=7, d=2):
        super(RepCSD, self).__init__()

        self.DOWN0 = MFF_Block(in_channels=4, out_channels=num_channels_list[0], use_atten=False, deploy=deploy)
        self.DOWN1 = MFF_Block(in_channels=num_channels_list[0], out_channels=num_channels_list[1], use_atten=False, deploy=deploy)
        self.DOWN2 = MFF_Block(in_channels=num_channels_list[1], out_channels=num_channels_list[2], use_atten=False, deploy=deploy)
        self.DOWN3 = MFF_Block(in_channels=num_channels_list[2], out_channels=num_channels_list[3], use_atten=False, dila=2, deploy=deploy)
        # self.DOWN4 = MFF_Block(in_channels=num_channels_list[3], out_channels=num_channels_list[4], use_atten=use_atten, dila=4, deploy=deploy)

        # self.attenC = ECA_Layer(5)
        # self.attenS = SpatialAtten2(kernel_size=5, size=128)
        # self.attenS = SpatialAtten(2, 1, 7)

        self.UP0 = MFF_Block(k=k, d=d, in_channels=num_channels_list[3], out_channels=num_channels_list[3], use_atten=True, is_down=False, dila=4, deploy=deploy, is_transp=True)
        self.UP1 = MFF_Block(k=k, d=d, in_channels=num_channels_list[3], out_channels=num_channels_list[2], use_atten=True, is_down=False, deploy=deploy, is_transp=True, size=128)
        self.UP2 = MFF_Block(k=k, d=d, in_channels=num_channels_list[2], out_channels=num_channels_list[1], use_atten=True, is_down=False, deploy=deploy, is_transp=True, size=256)
        self.UP3 = MFF_Block(k=k, d=d, in_channels=num_channels_list[1], out_channels=num_channels_list[0], use_atten=True, is_down=False, deploy=deploy, is_transp=True, size=512)

        self.OUT = OUT_Block(in_channels=num_channels_list[0], mid_channels=8, out_channels=num_classes, use_atten=False)

        self.DS0 = DS_Block(in_channels=num_channels_list[3], mid_channels=2, out_channels=1, kernel_size=16, stride=8, padding=4, use_atten=False)
        self.DS1 = DS_Block(in_channels=num_channels_list[2], mid_channels=2, out_channels=1, kernel_size=8, stride=4, padding=2, use_atten=False)
        self.DS2 = DS_Block(in_channels=num_channels_list[1], mid_channels=2, out_channels=1, kernel_size=4, stride=2, padding=1, use_atten=False)

    def forward(self, x):
        ski0, down0 = self.DOWN0(x)
        ski1, down1 = self.DOWN1(down0)
        ski2, down2 = self.DOWN2(down1)
        ski3, down3 = self.DOWN3(down2)
        # ski4, down4 = self.DOWN4(down3)

        # c = self.attenC(ski3)
        # s = self.attenS(c)
        # up0 = self.UP0(c)

        up0 = self.UP0(ski3)
        # up0 = self.UP0(ski4, ski3)
        # up1 = self.UP1(ski3, ski2)
        up1 = self.UP1(up0, ski2)
        up2 = self.UP2(up1, ski1)
        up3 = self.UP3(up2, ski0)

        # m0, ds0 = self.DS0(ski3)
        m0, ds0 = self.DS0(up0)
        m1, ds1 = self.DS1(up1)
        m2, ds2 = self.DS2(up2)

        # out = self.OUT(up3)
        out = self.OUT(up3, m0, m1, m2)

        return [ds0, ds1, ds2, out]
        # return out
