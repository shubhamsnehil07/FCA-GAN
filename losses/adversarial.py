import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.loss(preds, targets)
