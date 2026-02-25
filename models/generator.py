import torch
import torch.nn as nn
from .attention import SpatialAttention
from .frequency import FrequencyExtractor


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        # Encoder
        self.down1 = UNetBlock(in_channels, 64, use_bn=False)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)

        # Frequency Module
        self.freq = FrequencyExtractor(512)

        # Attention
        self.attn = SpatialAttention()

        # Decoder
        self.up1 = UNetBlock(512, 256, down=False)
        self.up2 = UNetBlock(512, 128, down=False)
        self.up3 = UNetBlock(256, 64, down=False)
        self.final = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Frequency fusion
        freq_feat = self.freq(d4)
        d4 = d4 + freq_feat

        # Attention
        d4 = self.attn(d4)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        out = self.final(torch.cat([u3, d1], dim=1))

        return self.tanh(out)
