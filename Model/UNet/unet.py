import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(inner_channels),
        nn.GELU(),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
    )
    return down_conv

# def x2conv(in_channels, out_channels, inner_channels=None):
#     inner_channels = out_channels // 2 if inner_channels is None else inner_channels
#     down_conv = nn.Sequential(
#         nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(inner_channels),
#         nn.ReLU(inplace=True),
#         # nn.GELU(),
#         nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         # nn.GELU()
#         nn.ReLU(inplace=True)
#     )
#     return down_conv

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
    def __init__(self, num_classes, in_channels=4, freeze_bn=False):
        super(UNet, self).__init__()

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
    def _initialize_weights(self):
        num = 0
        for m in self.modules():
            num +=1
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        print(f"depth: {num}")
    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))
        # x = self.dropout(x)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        #print(f"x: {x.shape}")
#         x = self.softmax(x)

        return x





