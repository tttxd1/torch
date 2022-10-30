import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from torch.nn import init


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.GELU(),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
    )
    return down_conv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3)):
            if interpolate:
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode="bilinear", align_corners=True)
            else:
                diffy = x_copy.size()[2] - x.size()[2]
                diffx = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffx//2, diffx - diffx//2),
                              diffy//2, diffy - diffy//2)

        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=4, image_size=512 , freeze_bn=False):
        super(UNet, self).__init__()

        #位置信息
        self.embedding = nn.Embedding(100, image_size*image_size)
        # self.linear0 = nn.Linear(in_features=2*image_size*image_size, out_features=image_size*image_size)
        self.conv11 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)




        #Unet
        self.start_conv = x2conv(in_channels, 64)
        self.down1 = Encoder(64, 128)
        self.down2 = Encoder(128, 256)
        self.down3 = Encoder(256, 512)
        self.down4 = Encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

        self.dropout = nn.Dropout(p=0.5)

        if freeze_bn:
            self.freeze_bn()
    # def _initialize_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #             nn.init.kaiming_normal_(module.weight)
    #             if module.bias is not None:
    #                 module.bias.data.zero_()
    #             elif isinstance(module, nn.BatchNorm2d):
    #                 module.weight.data.fill_(1)
    #                 module.bias.data.zero_()

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                init.xavier_uniform_(m.weight, gain=1)
                # print(m.weight)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, postion):

        #Postion Embedding
        postion_embeding = self.embedding(postion)  #[2, 512*512]
        postion_embeding = postion_embeding.reshape((2,512, 512))
        postion_embeding = self.conv11(postion_embeding)
        # postion_embeding = self.linear0(postion_embeding) #[1, 512*512]
        # postion_embeding = postion_embeding.reshape((512,512))
        # print(postion_embeding.shape)

        x = x
        #Unet
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))
        x = self.dropout(x)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        #print(f"x: {x.shape}")
        x = self.softmax(x)

        return x





