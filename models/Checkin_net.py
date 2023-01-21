import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from Vector_net import Vec_net
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
from block import fusions
from models.ResNextBlock import ResNextBlock


class Checkin_net(nn.Module):
    def __init__(self, in_channel, out_channel, in_num, out_dim):
        super(Checkin_net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_num=in_num
        # self.group_num = 32
        self.out_dim=out_dim
        self.vec_module=nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(inplace=True),
        )
        self.resnext_module_1 = ResNextBlock(256, 128)
        self.resnext_module_2 = ResNextBlock(256, 128)

        self.linear=nn.Linear(self.out_channel*self.in_num, self.out_dim)

    def forward(self, Checkin):
        x1=self.vec_module(Checkin)
        # print(x1.shape)  Checkin_net(120, 256, 1, 64)
        x2=self.resnext_module_1(x1)
        # print(x2.shape)
        # x2=self.vec_net2(x1)
        x3=self.resnext_module_2(x2)
        shortcut = x1+x3

        shortcut=shortcut.view(shortcut.size(0), -1)
        # print(shortcut.shape)
        return self.linear(shortcut)

if __name__ == '__main__':
    x = torch.randn(64, 120, 1)
    out = Checkin_net(120, 256, 1, 2)(x)
    print(out.shape)
