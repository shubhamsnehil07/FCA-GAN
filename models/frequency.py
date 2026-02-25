import torch
import torch.nn as nn


class FrequencyExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        fft = torch.fft.fft2(x)
        magnitude = torch.abs(fft)
        return self.conv(magnitude)
