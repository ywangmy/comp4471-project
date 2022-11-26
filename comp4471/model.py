import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def get_pretrained(num_classes = 1000):
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    efficientnet = torchvision.models.efficientnet_v2_s(weights='DEFAULT', progress=True, num_classes=num_classes) # 150MB, 20M params for 1000 classes
    # efficientnet = torchvision.models.efficientnet_v2_l(weights='DEFAULT', progress=True, num_classes=num_classes) # 455M, 118M params for 1000 classes
    return efficientnet

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = get_pretrained()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return efficientnet(x)

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model = model.to(device)