# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/9/13 19:42 
@Author : 弓长广文武
======================================
"""
from collections import OrderedDict
from math import log

import torch
from torch import nn
import numpy as np


from Rep_code.CBAM import SpatialAtten, SpatialAtten1, SpatialAtten2, ChannelAtten
from Rep_code.ECANet import ECA_Layer, ECA_Layer1
from Rep_code.NonLocal import NonLocal
from Rep_code.SENet import SE_Block

'''
======================================
@File    :   RepCSD_utils.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
# class Conv_ReLU(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
#         super(Conv_ReLU, self).__init__()
#         self.out = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)),
#                     ("relu", nn.ReLU(inplace=True))
#                 ]
#             )
#         )
#     def forward(self, x):
#         return self.out(x)

# class Conv_Bn_ReLU(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
#         super(Conv_Bn_ReLU, self).__init__()
#         self.out = nn.Sequential(
#             OrderedDict(
#                 [
#                     ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)),
#                     ('bn', nn.BatchNorm2d(out_channels)),
#                     ('relu', nn.ReLU(inplace=True))
#                 ]
#             )
#         )
#
#     def forward(self, x):
#         return self.out(x)


# class Conv_Bn(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
#         super(Conv_Bn, self).__init__()
#         self.out = nn.Sequential()
#         self.out.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.out.add_module('bn', nn.BatchNorm2d(out_channels))
#
#     def forward(self, x):
#         return self.out(x)


def Conv_ReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):

    out = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)),
                ("relu", nn.ReLU(inplace=True))
            ]
        )
    )
    return out

def Conv_Bn_ReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):

    out = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU(inplace=True))
            ]
        )
    )
    return out

def Conv_Bn(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):

    out = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)),
                ("bn", nn.BatchNorm2d(out_channels)),
            ]
        )
    )
    return out

def Conv(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):

    out = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)),
            ]
        )
    )
    return out

class TransposeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1):
        super(TransposeConv, self).__init__()
        self.out = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.out(x)


class MFF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dila=1, deploy=False, use_atten=False, groups=1, is_down=True, is_upsam=False, is_transp=False, size=128, k=7, d=2):
        super(MFF_Block, self).__init__()

        self.deploy = deploy  # 是否部署
        self.is_down = is_down  # 是否是下采样
        self.is_upsam = is_upsam
        self.is_transp = is_transp
        self.groups = groups
        self.in_channels = in_channels  # 输入通道
        self.out_channels = out_channels  # 输出通道
        self.nonlinear = nn.ReLU()  # 非线性激活
        self.size = size

        # self.bias_conv = 0

        if use_atten:  # 该block是否使用注意力
            # self.attenC = SE_Block(in_channels)
            # self.atten = ECA_Layer(in_channels, 5)
            # self.ks_eca = int(abs(log(self.in_channels, 2)/2+1/2))
            # self.ks_eca = int(round(((256 - self.in_channels) // 64)))
            # if self.in_channels == 256:
            #     self.ks_eca = 7
            # elif self.in_channels == 128:
            #     self.ks_eca = 5
            # else:
            #     self.ks_eca = 5
            self.attenC = ECA_Layer(5)
            # self.attenC = ECA_Layer1(5)
            # self.attenC = ChannelAtten(in_channels, 16)
            # self.attenC = ChannelAttenModule()
            # self.attenS = PositionAttenModule(in_channels)
            self.attenS = SpatialAtten(2, 1, k, d)
            # self.attenS = SpatialAtten1(in_channels, in_channels//2, 3, 2)
            # self.attenS = SpatialAtten2(kernel_size=5, size=self.size)
            # self.atten = NonLocal(out_channels)
        else:  # 不使用注意力的话直接传统方式进行映射
            self.attenC = nn.Identity()
            self.attenS = nn.Identity()

        # 判断是进行下采样还是上采样
        if self.is_upsam:  # 上采样就进行转职卷积或者上采样
            self.upsam = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            )
        elif self.is_transp:  # 上采样就进行转职卷积或者上采样
            self.upsam = TransposeConv(in_channels=in_channels, out_channels=out_channels,  kernel_size=4,
                                       stride=2, padding=1, dilation=1)
        elif self.is_down:  # 下采样就进行池化，或者是在dense中进行下采样
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            assert self.is_down


        # 如果进行部署
        if self.deploy:
            self.Re_Para0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dila, dilation=dila)
            self.Re_Para1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dila, dilation=dila)
            self.Re_Para2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dila, dilation=dila)
            # self.Re_Para0 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dila, dilation=dila, groups=1, bias=False)
            # # self.Bn0 = nn.BatchNorm2d(num_features=out_channels)
            # self.Re_Para1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dila, dilation=dila, groups=1, bias=True)
        else:
            # 并行3*3卷积块
            self.P_Conv3 = Conv_Bn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                        padding=dila, dilation=dila)
            # 并行1*1卷积块
            self.P_Conv11 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                        padding=0, dilation=1)
            self.P_Conv21 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                        padding=0, dilation=1)
            # 并行1*1卷积块
            self.F_Conv01 = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                        padding=0, dilation=1)
            # 不做处理直接映射
            self.Iden = nn.Identity()
            # # 并行BN层处理映射
            # self.Iden = nn.BatchNorm2d(out_channels)

            # 第一个3*3卷积块
            self.F_Conv03 = Conv_Bn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=dila,
                                    dilation=dila)
            # 最后的1*1卷积块
            self.L_Conv1 = Conv_Bn_ReLU(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1,
                                        padding=0, dilation=1)

            # 并行双3*3卷积块
            self.P_Conv13 = Conv_Bn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=dila, dilation=dila)
            self.P_Conv23 = Conv_Bn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=dila, dilation=dila)

    def forward(self, x, s=None):
        if self.deploy:  # 部署的话
            if not self.is_down:
                if s is None:
                    cat1 = x
                else:
                    up = self.upsam(x)
                    cat1 = torch.cat((up, s), dim=1)
                    cat1 = self.attenC(cat1)
                    cat1 = self.attenS(cat1)
                Re0 = self.nonlinear(self.Re_Para0(cat1))
                # FC03_R = self.nonlinear(FC03)
                Re1 = self.nonlinear(self.Re_Para1(Re0))
                Re2 = self.nonlinear(self.Re_Para2(Re1))
                out = Re2

                return out
            else:  # 如果是编码部分
                # d03 = self.dense03(x)
                # d11 = self.sparse11(d03)
                # d13 = self.dense13(d03)
                # d23 = self.dense23(d03)
                # cat = torch.cat((d11, d13, d23), dim=1)
                # d01 = self.sparse01(cat)
                # out = self.pool(d01)
                # return d01, out

                Re0 = self.nonlinear(self.Re_Para0(x))
                # RP0 = self.nonlinear(self.Re_Para0(FC03))
                Re1 = self.nonlinear(self.Re_Para1(Re0))
                Re2 = self.nonlinear(self.Re_Para2(Re1))
                # ski = self.atten(RP0 + PC23)
                ski = Re2
                out = self.pool(ski)
                return ski, out


        else:  # 如果不是部署
            if not self.is_down:  # 如果是解码部分
                # if not s:
                #     cat1 = x
                # else:
                # up = self.upsam(x)
                # cat1 = torch.cat((up, s), dim=1)
                # u03 = self.dense03(cat1)
                # u11 = self.sparse11(u03)
                # u13 = self.dense13(u03)
                # u23 = self.dense23(u03)
                # cat2 = torch.cat((u11, u13, u23), dim=1)
                # out = self.sparse01(cat2)

                if s is None:
                    cat = x
                else:
                    up = self.upsam(x)
                    cat = torch.cat((up, s), dim=1)
                    attenC = self.attenC(cat)
                    attenS = self.attenS(attenC)
                    cat = attenS
                    # cat = attenS * attenC * cat
                FC03 = self.F_Conv03(cat)
                FC01 = self.F_Conv01(cat)
                FC0 = self.nonlinear(FC03 + FC01)
                # FC0 = self.nonlinear(FC03)
                # FC = self.attenS(FC)
                PC13 = self.P_Conv13(FC0)
                PC11 = self.P_Conv11(FC0)
                Iden1 = self.Iden(FC0)
                PC1 = self.nonlinear(PC13 + PC11 + Iden1)
                # PC1 = self.nonlinear(PC13)
                # PC0 = self.attenS(PC0)
                PC23 = self.P_Conv23(PC1)
                PC21 = self.P_Conv21(PC1)
                Iden2 = self.Iden(PC1)
                PC2 = self.nonlinear(PC23 + PC21 + Iden2)
                # PC2 = self.nonlinear(PC23)
                out = PC2

                # PC3 = self.P_Conv3(FC3)
                # PC23 = self.P_Conv13(self.P_Conv03(FC3))
                # PC = PC23 + self.nonlinear(PC3 + PC1 + Iden)
                # Iden = self.Iden(FC3)
                # Iden1 = self.Iden1(PC0)
                # PC11 = self.P_Conv11(PC0)
                # PC13 = self.P_Conv13(PC0)
                # PC1 = self.nonlinear(PC13 + PC11 + Iden1)
                # PC23 = self.P_Conv13(self.P_Conv03(FC3))
                # out = self.atten(Iden + PC1 + PC23)
                # PC13 = self.P_Conv13(PC03)
                # out = self.atten(PC)
                # out = self.atten(self.nonlinear(PC13 + PC1 + Iden))

                # return outFC3 = self.F_Conv3(cat1)

                return out
                # return PC1

            else:  # 如果是编码部分
                # d03 = self.dense03(x)
                # d11 = self.sparse11(d03)
                # d13 = self.dense13(d03)
                # d23 = self.dense23(d03)
                # cat = torch.cat((d11, d13, d23), dim=1)
                # d01 = self.sparse01(cat)
                # out = self.pool(d01)
                # return d01, out

                # FC3 = self.F_Conv3(x)
                # # Iden = self.Iden(FC3)
                # # PC1 = self.P_Conv1(FC3)
                # # PC3 = self.P_Conv3(FC3)
                # PC23 = self.P_Conv13(self.P_Conv03(FC3))
                # # PC = PC23 + self.nonlinear(PC3 + PC1 + Iden)
                # # Iden0 = self.Iden0(FC3)
                # # PC01 = self.P_Conv01(FC3)
                # # PC03 = self.P_Conv03(FC3)
                # # PC0 = self.nonlinear(PC03 + PC01 + Iden0)
                # # Iden1 = self.Iden1(PC0)
                # # PC11 = self.P_Conv11(PC0)
                # # PC13 = self.P_Conv13(PC0)
                # # PC1 = self.nonlinear(PC13 + PC11 + Iden1)
                # # Iden = self.Iden(FC3)
                # # PC1 = self.P_Conv1(FC3)
                # # # PC23 = self.P_Conv13(self.P_Conv03(FC3))
                # # # ski = self.atten(Iden + PC1 + PC23)
                # # PC03 = self.P_Conv03(FC3)
                # # # PC13 = self.P_Conv13(PC03)
                # # ski = self.atten(PC)
                # # ski = self.atten(self.nonlinear(PC13 + PC1 + Iden))
                # # out = self.pool(ski)
                FC03 = self.F_Conv03(x)
                FC01 = self.F_Conv01(x)
                FC0 = self.nonlinear(FC03 + FC01)
                # FC0 = self.nonlinear(FC03)
                PC13 = self.P_Conv13(FC0)
                PC11 = self.P_Conv11(FC0)
                Iden1 = self.Iden(FC0)
                PC1 = self.nonlinear(PC13 + PC11 + Iden1)
                # PC1 = self.nonlinear(PC13)
                PC23 = self.P_Conv23(PC1)
                PC21 = self.P_Conv21(PC1)
                Iden2 = self.Iden(PC1)
                PC2 = self.nonlinear(PC23 + PC21 + Iden2)
                # PC2 = self.nonlinear(PC23)
                ski = PC2

                out = self.pool(ski)
                return ski, out
                # return PC1, out

    # 获取经过计算所得到的weights和bias
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # elif isinstance(branch, nn.Module):
            weight = branch.conv.weight  # 卷积权重
            bias = branch.conv.bias  # 卷积系数
            if hasattr(branch, 'bn'):
                running_mean = branch.bn.running_mean  # bn均值
                running_var = branch.bn.running_var  # bn均方根
                gamma = branch.bn.weight  # 重构变换参数 bn权重
                beta = branch.bn.bias  # bn偏移系数
                eps = branch.bn.eps  # bn var + eps
                num_batches_tracked = branch.bn.num_batches_tracked

                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                # weight = (weight.permute(1, 0, 2, 3) * t).permute(1, 0, 2, 3)
                return weight * t, beta + (bias - running_mean) * gamma / std
            else:
                return weight, bias

        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.out_channels // self.groups
                kernel_value = np.zeros((self.out_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.out_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            weight = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

            # return weight, bias
            # return weight, bias, running_mean, running_var, gamma, beta, eps, num_batches_tracked

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            # weight = (weight.permute(1, 0, 2, 3) * t).permute(1, 0, 2, 3)
            return weight * t, beta + ( - running_mean) * gamma / std
            # return weight * t, beta - running_mean * gamma / std
            # return weight, bias

    # 获取w与b，以可微的方式导出等价的权重与偏差
    def get_equivalent_weights_bias(self):
        # 第一层
        w_F_Conv03, b_F_Conv03 = self._fuse_bn_tensor(self.F_Conv03)
        w_F_Conv01, b_F_Conv01 = self._fuse_bn_tensor(self.F_Conv01)
        # 第二层
        w_P_Conv13, b_P_Conv13 = self._fuse_bn_tensor(self.P_Conv13)
        w_P_Conv11, b_P_Conv11 = self._fuse_bn_tensor(self.P_Conv11)
        w_Iden1 = torch.zeros(size=list(w_P_Conv13.size()), dtype=torch.float32)
        input_dim = self.out_channels // self.groups
        for i in range(self.out_channels):
            w_Iden1[i, i % input_dim, 1, 1] = 1
        # 第三层
        w_P_Conv23, b_P_Conv23 = self._fuse_bn_tensor(self.P_Conv23)
        w_P_Conv21, b_P_Conv21 = self._fuse_bn_tensor(self.P_Conv21)
        w_Iden2 = torch.zeros(size=list(w_P_Conv23.size()), dtype=torch.float32)
        input_dim = self.out_channels // self.groups
        for i in range(self.out_channels):
            w_Iden2[i, i % input_dim, 1, 1] = 1

        return self._pad_1x1_to_3x3_tensor(w_F_Conv01) + w_F_Conv03, b_F_Conv03 + b_F_Conv01\
            , w_Iden1 + self._pad_1x1_to_3x3_tensor(w_P_Conv11) + w_P_Conv13, b_P_Conv13 + b_P_Conv11\
            , w_Iden2 + self._pad_1x1_to_3x3_tensor(w_P_Conv21) + w_P_Conv23, b_P_Conv23 + b_P_Conv21


        # weight, bias, running_mean, running_var, gamma, beta, eps, num_batches_tracked = self._fuse_bn_tensor(self.P_Conv03)
        # weight, bias = self._fuse_bn_tensor(self.P_Conv03)
        # w_F_Conv3, b_F_Conv3 = self._fuse_bn_tensor(self.F_Conv3)
        # w_P_Conv1, b_P_Conv1 = self._fuse_bn_tensor(self.P_Conv1)
        # w_Iden, b_Iden = torch.ones(size=list(w_P_Conv1.shape), dtype=torch.float32), torch.zeros(size=list(b_P_Conv1.size()), dtype=torch.float32)
        # w_P_Conv3, b_P_Conv3 = self._fuse_bn_tensor(self.P_Conv3)
        # w_P_Conv1 = self._fuse_bn_tensor(self.P_Conv1)
        # w_P_Conv3 = self._fuse_bn_tensor(self.P_Conv3)
        # w_Iden = torch.zeros(size=list(w_P_Conv3.size()), dtype=torch.float32)
        # input_dim = self.out_channels // self.groups
        # for i in range(self.out_channels):
        #     w_Iden[i, i % input_dim, 1, 1] = 1
        # w_P_Conv03, b_P_Conv03 = self._fuse_bn_tensor(self.P_Conv03)
        # w_P_Conv13, b_P_Conv13 = self._fuse_bn_tensor(self.P_Conv13)
        # return w_Iden + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_P_Conv3
        # return (w_P_Conv13 * w_P_Conv03) + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_Iden, b_P_Conv13 + (w_P_Conv13 * b_P_Conv03.reshape(-1, 1, 1, 1)) + b_P_Conv1 + b_Iden
        # return (w_P_Conv13 * w_P_Conv03) + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_Iden, b_P_Conv13 + b_P_Conv1 + b_Iden
        # return (w_P_Conv13 * w_P_Conv03), b_P_Conv13
        # return w_P_Conv03, w_P_Conv13, b_P_Conv13
        # return (w_P_Conv13 * w_P_Conv03) + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_Iden, (w_P_Conv13 * b_P_Conv03.reshape(-1, 1, 1, 1))
        # return weight, bias, running_mean, running_var, gamma, beta, eps, num_batches_tracked
        # return weight, bias
        # w_P_Conv13, b_P_Conv13,
        # return w_P_Conv03 + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_Iden, b_P_Conv03 + b_P_Conv1 + b_Iden
        #        w_P_Conv13 + self._pad_1x1_to_3x3_tensor(w_P_Conv1) + w_Iden, b_P_Conv13 + b_P_Conv1 + b_Iden,
        # return w_P_Conv03 + self._pad_1x1_to_3x3_tensor(w_P_Conv1), b_P_Conv03 + b_P_Conv1

    # 将1x1大小的卷积膨胀微3x3
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])  # 扩展数据大小

    # 选择部分层去重构网络
    def switch_to_deploy(self):
        if hasattr(self, 'Re_Para'):  # 如果本身就是重构的代码，那么就不在进行重构
            return
        w0, b0, w1, b1, w2, b2 = self.get_equivalent_weights_bias()
        # weights0, bias0, weights1, bias1, = self.get_equivalent_weights_bias()
        # weight = self.get_equivalent_weights_bias()
        # weight, bias, running_mean, running_var, gamma, beta, eps, num_batches_tracked = self.get_equivalent_weights_bias()
        # self.Re_Para0 = Conv_Bn(in_channels=self.P_Conv03.conv.in_channels, out_channels=self.P_Conv03.conv.out_channels,
        #                           kernel_size=self.P_Conv03.conv.kernel_size, stride=self.P_Conv03.conv.stride,
        #                           padding=self.P_Conv03.conv.padding, dilation=self.P_Conv03.conv.dilation)
        # self.Bn0 = nn.BatchNorm2d(self.P_Conv03.bn.num_features)
        self.Re_Para0 = nn.Conv2d(in_channels=self.F_Conv03.conv.in_channels, out_channels=self.F_Conv03.conv.out_channels,
                                  kernel_size=self.F_Conv03.conv.kernel_size, stride=self.F_Conv03.conv.stride,
                                  padding=self.F_Conv03.conv.padding, dilation=self.F_Conv03.conv.dilation,
                                  groups=self.F_Conv03.conv.groups, bias=True
                                  )
        self.Re_Para1 = nn.Conv2d(in_channels=self.P_Conv13.conv.in_channels, out_channels=self.P_Conv13.conv.out_channels,
                                  kernel_size=self.P_Conv13.conv.kernel_size, stride=self.P_Conv13.conv.stride,
                                  padding=self.P_Conv13.conv.padding, dilation=self.P_Conv13.conv.dilation,
                                  groups=self.P_Conv13.conv.groups, bias=True
                                  )
        self.Re_Para2 = nn.Conv2d(in_channels=self.P_Conv23.conv.in_channels, out_channels=self.P_Conv23.conv.out_channels,
                                  kernel_size=self.P_Conv23.conv.kernel_size, stride=self.P_Conv23.conv.stride,
                                  padding=self.P_Conv23.conv.padding, dilation=self.P_Conv23.conv.dilation,
                                  groups=self.P_Conv23.conv.groups, bias=True
                                  )
        self.Re_Para0.weight.data = w0
        self.Re_Para0.bias.data = b0
        self.Re_Para1.weight.data = w1
        self.Re_Para1.bias.data = b1
        self.Re_Para2.weight.data = w2
        self.Re_Para2.bias.data = b2
        # self.Re_Para0.bias.data = bias
        # self.Re_Para1.weight.data = weight
        # self.Re_Para1.bias.data = bias
        # self.conv_bias = conv_bias
        # self.Re_Para0.bn.running_mean = running_mean
        # self.Re_Para0.bn.running_var = running_var
        # self.Re_Para0.bn.weight = gamma
        # self.Re_Para0.bn.bias = beta
        # self.Re_Para0.bn.eps = eps
        # self.Re_Para0.bn.num_batches_tracked = num_batches_tracked

        # self.Re_Para1.weight.data = weights1
        # self.Re_Para1.bias.data = bias1
        for para in self.parameters():
            para.detach_()  # 类似于深拷贝，并且使得变量没有梯度
        self.__delattr__('P_Conv13')  # 删除没用的分支
        self.__delattr__('P_Conv11')  # 删除没用的分支
        self.__delattr__('P_Conv23')  # 删除没用的分支
        self.__delattr__('P_Conv21')  # 删除没用的分支
        self.__delattr__('F_Conv03')  # 删除没用的分支
        self.__delattr__('F_Conv01')  # 删除没用的分支
        self.__delattr__('L_Conv1')  # 删除没用的分支
        self.__delattr__('P_Conv3')  # 删除没用的分支
        if hasattr(self, 'Iden'):
            self.__delattr__('Iden')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class OUT_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_atten=False):
        super(OUT_Block, self).__init__()
        if out_channels < 3:
            out_channels = 1
        if use_atten:  # 该block是否使用注意力
            # self.atten = SE_Block(8)
            # self.atten = ECA_Layer(8, 5)
            self.atten = NonLocal(mid_channels)
        else:  # 不使用注意力的话直接传统方式进行映射
            self.atten = nn.Identity()
        self.dense = Conv_Bn_ReLU(in_channels, 2, kernel_size=3, padding=1)
        # self.sparse = nn.Conv2d(2, out_channels, kernel_size=1)
        self.sparse = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x, s5=None, s6=None, s7=None):
        o = self.dense(x)
        # cat = (torch.cat((s5, s6, s7, o), dim=1))
        if s5 == None or s6 == None or s7 == None:
            out = self.sparse(o)
        else:
            cat = self.atten(torch.cat((s5, s6, s7, o), dim=1))
            out = self.sparse(cat)
        return out


class DS_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deploy=False, use_atten=False, is_upsam=False):
        super(DS_Block, self).__init__()
        if use_atten:
            self.atten = SE_Block(in_channels)
        else:
            self.atten = nn.Identity()
        if is_upsam:
            self.upsam = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1)),
            )
        else:
            self.upsam = TransposeConv(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation)
        self.dense = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                padding=1, dilation=1)
        self.sparse = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                padding=0, dilation=1)

    def forward(self, x):
        x = self.atten(x)
        up = self.upsam(x)
        out = self.sparse(up)
        # out = self.dense(up)
        return up, out