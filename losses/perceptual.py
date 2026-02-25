import torch
import torch.nn.functional as F
from torchvision.models import vgg16

class PerceptualLoss:
    def __init__(self, device):
        self.vgg = vgg16(pretrained=True).features.to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        self.selected_layers = [3, 8]

    def __call__(self, fake, real):
        f, r = (fake + 1) / 2, (real + 1) / 2
        loss = 0.0
        x, y = f, r
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.selected_layers:
                loss += F.mse_loss(x, y)
        return loss
