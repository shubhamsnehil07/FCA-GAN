import torch
import torch.nn as nn
import torch.nn.functional as F
from .frequency import get_dct_weights

class MultiSpectralChannelAttention(nn.Module):
    def __init__(self, channel, height, width, reduction=16, num_freq=16):
        super().__init__()
        self.num_freq = num_freq
        self.height = height
        self.width = width
        self.register_buffer('dct_weights', get_dct_weights(height, width, channel, num_freq))

        self.fc = nn.Sequential(
            nn.Linear(num_freq * channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        current_dct = F.interpolate(self.dct_weights, size=(h, w), mode='bilinear') if (h != self.height or w != self.width) else self.dct_weights
        
        outputs = []
        for i in range(self.num_freq):
             freq_w = current_dct[:, i, :, :].unsqueeze(0)
             component = (x * freq_w).sum([2, 3])
             outputs.append(component)

        dct_feats = torch.cat(outputs, dim=1)
        att = self.fc(dct_feats).view(b, c, 1, 1)
        return x * att

class FCA_Block(nn.Module):
    def __init__(self, channel, input_shape):
        super().__init__()
        self.fca = MultiSpectralChannelAttention(channel, input_shape[0], input_shape[1], reduction=16, num_freq=9)
        self.spatial = nn.Conv2d(channel, 1, kernel_size=7, padding=3)

    def forward(self, x):
        out = self.fca(x)
        sp_att = torch.sigmoid(self.spatial(out))
        return out * sp_att
