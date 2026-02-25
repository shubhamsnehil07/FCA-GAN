import torch
import torch.nn as nn
from .attention import FCA_Block

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, use_attention=False, feature_hw=None):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.use_attention = use_attention
        self.model = nn.Sequential(*layers)
        if use_attention and feature_hw is not None:
             self.att = FCA_Block(out_size, feature_hw)

    def forward(self, x):
        x = self.model(x)
        if self.use_attention: x = self.att(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, use_attention=False, feature_hw=None):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False), nn.BatchNorm2d(out_size)]
        if dropout: layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.use_attention = use_attention
        self.model = nn.Sequential(*layers)
        if use_attention and feature_hw is not None:
            self.att = FCA_Block(out_size, feature_hw)

    def forward(self, x, skip=None):
        x = self.model(x)
        if self.use_attention: x = self.att(x)
        if skip is not None: x = torch.cat((x, skip), dim=1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128, use_attention=True, feature_hw=(64,64))
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, use_attention=True, feature_hw=(16,16))
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5, use_attention=True, feature_hw=(2,2))
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5, use_attention=True, feature_hw=(8,8))
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256, use_attention=True, feature_hw=(32,32))
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64, use_attention=True, feature_hw=(128,128))

        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8, d7); u2 = self.up2(u1, d6); u3 = self.up3(u2, d5); u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3); u6 = self.up6(u5, d2); u7 = self.up7(u6, d1)
        return self.final(u7)
