import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.abs(torch.fft.fft2(pred))
        target_fft = torch.abs(torch.fft.fft2(target))
        return self.l1(pred_fft, target_fft)
