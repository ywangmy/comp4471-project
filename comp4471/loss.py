import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
class evalLoss(nn.BCELoss):
    def __init__(self):
        super().__init__(weight=None, reduction='sum')

class dynamicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y):
        return 0

class staticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perFrame = nn.BCELoss(weight=None, reduction='mean')

    def forward(self, pred, y):
        sum = self.perFrame(pred, y)
        return sum

class twoPhaseLoss(nn.Module):
    def __init__(self, phase, W_dynamic = 1.):
        super().__init__()
        self.static_loss = staticLoss()
        self.dynamic_loss = dynamicLoss()
        self.phase = phase
        self.W_dynamic = W_dynamic

    def forward(self, pred, y):
        if self.phase == 1:
            return self.static_loss(pred, y)
        else:
            return self.static_loss(pred, y) # + self.dynamic_loss() * self.W_dynamic