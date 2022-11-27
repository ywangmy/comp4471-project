import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def get_pretrained(num_classes = 1000):
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    efficientNet = torchvision.models.efficientnet_v2_s(weights='DEFAULT', progress=True, num_classes=num_classes) # 150MB, 20M params for 1000 classes
    # efficientNet = torchvision.models.efficientnet_v2_l(weights='DEFAULT', progress=True, num_classes=num_classes) # 455M, 118M params for 1000 classes
    return efficientNet

class dynamicLoss(nn.Module):
    def __init__(self):
        pass

class ASRID(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientNet = get_pretrained()
        self.MAT = None
        self.static = None
        self.dynamic = None
        self.w = Tensor()

    def forward(self, x):
        x = self.efficientNet(x)
        att = self.MAT(x)
        x = self.static(att)
        y = self.dynamic(att)
        return self.w * x + (1. - self.w) * y