import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            block(64, 128, 2),
            block(128, 256, 2),
            block(256, 512, 1),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
