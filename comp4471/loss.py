import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
class evalLoss(nn.BCELoss):
    def __init__(self):
        super().__init__(weight=None, reduction='mean')

class dynamicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y)
        return 0

class staticLoss(nn.Module):
    def __init__(self):
        self.perFrame = evalLoss()

    def forward(self, pred, y)
        sum = self.perFrame(pred, y)
        return sum

class twoPhaseLoss(nn.Module):
    def __init__(self, W_dynamic = 1.):
        self.static_loss = staticLoss()
        self.dynamic_loss = dynamicLoss()
        self.W_dynamic = W_dynamic

    def forward(self, phase, pred, y)
        if phase == 1:
            return self.static_loss(pred, y)
        else:
            return self.static_loss(pred, y) # + self.dynamic_loss() * self.W_dynamic