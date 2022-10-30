#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
====================================================================================
PROJECT_NAME: building ; 
FILE_NAME: U-Net ;
AUTHOR: gbt ;
DATE: 2022/9/15 ;
CONTENT: unet网络;
====================================================================================
"""
import torch
from torch import nn
from torch.nn import functional as f


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)

    def forward(self, x, feature_map):
        up = f.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()
        self.c1 = ConvBlock(3, 64)
        self.d1 = DownSample(64)
        self.c2 = ConvBlock(64, 128)
        self.d2 = DownSample(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSample(256)
        self.c4 = ConvBlock(256, 512)
        self.d4 = DownSample(512)
        self.c5 = ConvBlock(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, num_class, 3, 1, 1)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        l1 = self.c1(x)
        l2 = self.c2(self.d1(l1))
        l3 = self.c3(self.d2(l2))
        l4 = self.c4(self.d3(l3))
        l5 = self.c5(self.d4(l4))
        r1 = self.c6(self.u1(l5, l4))
        r2 = self.c7(self.u2(r1, l3))
        r3 = self.c8(self.u3(r2, l2))
        r4 = self.c9(self.u4(r3, l1))
        return self.act(self.out(r4))


if __name__ == '__main__':
    x1 = torch.randn(8, 3, 256, 256)
    net = UNet(num_class=2)
    print(net(x1).shape)
