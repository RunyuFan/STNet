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
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from models.Checkin_net import Checkin_net
from models.vit import ViT
from models.transformer_block import Transformer1d
from models.MLPMixer import mlp_mixer_s16, mlp_mixer_b16, mlp_mixer_s32
# from models.resnext import resnext50
from torchsummary import summary
from models.resnet import ResNetFeature

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

segformer_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}

class ResMixer(nn.Module):
    def __init__(self, n_class):
        super(ResMixer,self).__init__()
        self.n_class=n_class

        self.resnet50_feature = ResNetFeature(128)
        print('resnet50_feature parameters:', sum(p.numel() for p in self.resnet50_feature.parameters() if p.requires_grad))

        self.mixer = mlp_mixer_s16(num_classes=64, image_size=64, channels = 128)
        self.fc = nn.Linear(64, self.n_class)

    def forward(self, img, msi, checkin):

        out = self.resnet50_feature(img)
        # out = channel_out + out
        out_feature = self.mixer(out)
        # out_feature = self.ViT(out)
        out = self.fc(out_feature)
        # print(out.shape)
        return out_feature, out

class Mixer_base(nn.Module):
    def __init__(self, n_class):
        super(Mixer_base,self).__init__()
        self.n_class=n_class

        # self.resnet50_feature = ResNetFeature(128)
        # print('resnet50_feature parameters:', sum(p.numel() for p in self.resnet50_feature.parameters() if p.requires_grad))

        self.mixer = mlp_mixer_s32(num_classes=2, image_size=256, channels = 3)
        # self.fc = nn.Linear(64, self.n_class)

    def forward(self, img, msi, checkin):

        # out = self.resnet50_feature(img)
        # out = channel_out + out
        out_feature = self.mixer(img)
        # out_feature = self.ViT(out)
        # out = self.fc(out_feature)
        # print(out.shape)
        return out_feature, out_feature

class Mixer(nn.Module):
    def __init__(self, n_class):
        super(Mixer,self).__init__()
        self.n_class=n_class

        resnext50 = models.resnext50_32x4d(pretrained=False)
        print('resnext50 parameters:', sum(p.numel() for p in resnext50.parameters() if p.requires_grad))
        self.resnext50 = list(resnext50.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnext50 = nn.Sequential(*self.resnext50)

        self.mixer = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        self.fc = nn.Linear(64, self.n_class)
        # self.ViT = ViT(
        #         image_size = 64,
        #         patch_size = 16,
        #         num_classes = 64,
        #         dim = 256,
        #         depth = 4,
        #         heads = 16,
        #         mlp_dim = 512,
        #         dropout = 0.1,
        #         emb_dropout = 0.1
        #     )

    def forward(self, img, msi, checkin):

        out = self.resnext50(img)
        # summary(self.resnext50, (3, 256, 256))
        # print(out.shape)

        # out1 = F.interpolate(x_layer1, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
        # out2 = F.interpolate(x_layer2, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
        # out3 = F.interpolate(x_layer3, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
        # out4 = F.interpolate(x_layer4, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)

        # out = torch.cat([out1, out2, out3, out4], 1)
        # print(out.shape)
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        # channel_out = self.ChannelAttentionModule(out)
        # out = channel_out + out
        out_feature = self.mixer(out)
        # out_feature = self.ViT(out)
        out = self.fc(out_feature)
        # print(out.shape)
        return out_feature, out

class STNet(nn.Module):
    def __init__(self, n_class):
        super(STNet,self).__init__()
        self.n_class=n_class

        self.img_model = torch.load('.\\model-UIS\\Mixer-6-2-2.pth')  # model trained with images
        # self.img_model.last_fc = nn.Dropout(0)

        self.Checkin_net_checkin = torch.load('.\\model-UIS\\CheckinNet-6-2-2.pth') # model trained with time series data
        # self.Checkin_net_checkin.last_fc = nn.Dropout(0)

        # self.fc = nn.Linear(4, 4)
        self.Transformer1d = Transformer1d(d_model=128, nhead=4)
        self.last_fc_out = nn.Linear(128, self.n_class)

    def forward(self, img, msi, checkin):
        feature_checkin, checkin = self.Checkin_net_checkin(img, msi, checkin)
        # checkin = checkin.unsqueeze(1)
        # msi = F.interpolate(msi, size=img.shape[-2:], mode='bilinear', align_corners=False)
        # ss_img = torch.cat([img, msi], 1)
        # ss_img = self.conv_block_5to3(ss_img)
        # print(img.shape, checkin.shape)
        feature_img, img = self.img_model(img, msi, checkin)
        # img = img.unsqueeze(1)

        # img = img.view(img.size(0), -1)
        # img = self.img_model_fc(img)
        # img = self.fc(img)

        # ss_img = self.img_model_msi(ss_img)
        # ss_img = ss_img.view(ss_img.size(0), -1)
        # ss_img = self.img_model_msi_fc(ss_img)
        # ss_img = self.fc(ss_img)

        # print(img.shape, ss_img.shape, checkin.shape)

        fuse_cat = torch.cat([feature_img, feature_checkin], 1)
        fuse_cat = fuse_cat.unsqueeze(1)
        # print(fuse_cat.shape)
        fuse_cat = self.Transformer1d(fuse_cat)
        # print(fuse_cat.shape)
        # fuse_cat = fuse_cat.squeeze(1)
        fuse_cat = fuse_cat.view(fuse_cat.size(0), -1)

        # fuse_cat = self.fc(fuse_cat)
        # c_w=nn.Sigmoid()(fuse_cat)

        # fuse_cat = c_w + fuse_cat

        out = self.last_fc_out(fuse_cat)
        return fuse_cat, out

class ImgNet(nn.Module):
    def __init__(self, n_class):
        super(ImgNet,self).__init__()
        self.n_class=n_class
        img_model = models.resnext101_32x8d(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d
        self.img_model = list(img_model.children())[:-2]
        self.img_model.append(nn.AdaptiveAvgPool2d(1))
        self.img_model = nn.Sequential(*self.img_model)


        self.fc = nn.Linear(img_model.fc.in_features, 64)
        self.last_fc = nn.Linear(64, self.n_class)


    def forward(self, img, msi, checkin):

        img = self.img_model(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)
        # fuse = torch.cat([checkin, img], 1)
        img_fc = self.last_fc(img)
        return img, img_fc

class CheckinNet(nn.Module):
    def __init__(self, n_class):
        super(CheckinNet, self).__init__()
        self.n_class = n_class
        self.Checkin_net_checkin = Checkin_net(120, 256, 1, 64)
        # self.fc = nn.Linear(120, 64)
        self.last_fc = nn.Linear(64, self.n_class)
    def forward(self, img, msi, checkin):
        # checkin = self.vit_checkin(checkin)
        # print(checkin.shape)
        # checkin = checkin.permute(0,2,1)
        checkin = self.Checkin_net_checkin(checkin)
        # checkin = self.fc(checkin)
        # print(checkin.shape)
        checkin_fc = self.last_fc(checkin)
        # print(checkin_fc.shape)
        return checkin, checkin_fc
