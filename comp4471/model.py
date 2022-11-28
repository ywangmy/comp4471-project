import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from MAT import MultiHeadAttention

def get_pretrained(num_classes = 1000):
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    efficientNet = torchvision.models.efficientnet_v2_s(weights='DEFAULT', progress=True, num_classes=num_classes) # 150MB, 20M params for 1000 classes
    # efficientNet = torchvision.models.efficientnet_v2_l(weights='DEFAULT', progress=True, num_classes=num_classes) # 455M, 118M params for 1000 classes
    return efficientNet

class MatNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        N, F, H, W = x.shape
        x = torch.linalg.matrix_norm(x) # https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
        x = torch.diff(x, dim=1) # https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff
        x = torch.abs(x)
        x = torch.mean(x, dim=1)
        return x

class staticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class ASRID(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientNet = get_pretrained()
        self.MAT = MultiHeadAttention(num_heads,input_size)
        self.static = staticClassifier()
        self.dynamic = MatNorm() # baseline

        # self.w_static = torch.rand((1,))
        self.w_static = 1.

    def forward(self, x):
        x = self.efficientNet(x)
        att = self.MAT(x)
        x = self.static(att).mean(dim=1)
        y = self.dynamic(att)
        return self.w_static * x + (1. - self.w_static) * y, att
