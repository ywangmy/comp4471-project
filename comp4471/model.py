import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .MAT import MultiHeadAttention
from .MAT import SelfAttention

def get_pretrained(num_classes = 1000):
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    if num_classes == 1000:
        efficientNet = torchvision.models.efficientnet_v2_s(weights='DEFAULT', progress=True) # 150MB, 20M params for 1000 classes
        # efficientNet = torchvision.models.efficientnet_v2_l(weights='DEFAULT', progress=True, num_classes=num_classes) # 455M, 118M params for 1000 classes
    else:
        efficientNet = torchvision.models.efficientnet_v2_s(num_classes=num_classes)
    return efficientNet

class MatNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        Input:
        - x: A feature vector (attention output), shape: (N, F, H, W)
        Output:
        - x: Dynamic score
        """
        #N, F, H, W = x.shape
        N, D = list(x.size())
        x = torch.linalg.matrix_norm(x) # https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
        x = torch.diff(x, dim=0) # https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff
        x = torch.abs(x)
        x = torch.mean(x, dim=1)
        return x

class staticClassifier(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        # FC as the final classification layer
        self.fc = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        """
        Input:
        - x (N, in_channels): A feature vector (attention ouput)
        Output:
        - score (N, out_channels=2): Static score
        """
        fc_output = self.fc(x)
        relu_output = F.relu(fc_output)
        score = F.softmax(relu_output)
        score = score[:, 0]
        return score

class ASRID(nn.Module):
    def __init__(self, batch_size, num_frames=1, num_features=1000, num_heads=10, dim_attn=1000):
        super().__init__()
        # Parameters setup
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_features = num_features
        self.num_heads = num_heads
        self.dim_attn = dim_attn

        # Blocks(/net/layer/model) setup
        self.efficientNet = get_pretrained(self.num_features)
        self.multiattn_block = SelfAttention(self.batch_size, self.num_frames, self.num_features, self.num_heads, self.dim_attn)
        self.static_block = staticClassifier(in_channels=self.dim_attn)
        self.dynamic_block = MatNorm() # baseline

        # Other parameters
        # self.w_static = torch.rand((1,))
        self.w_static = 1

    def forward(self, x):
        """
        Input:
        - x (N, F(rames), C(hannels), H, W): Raw information
        Output:
        - score (N, ): Score
        """
        # Change x: (N, F, ...) to x: (N * F, ...)
        #N, F, C, H, W = list(x.size())
        #print(f'x.size()={x.size()}, type={x.type()}')
        N, C, H, W = list(x.size())
        x = torch.reshape(x, (-1, C, H, W))

        # Feature output (N, num_features)
        feat_output = self.efficientNet(x)
        #print(f'feat_output.size()={feat_output.size()}')

        # Attention output (N, dim_attn * num_heads)
        attn_output, attn_output_weights = self.multiattn_block(feat_output)
        #print(f'attn_output.size()={attn_output.size()}')
        # Static (N, ) and Dynamic Block
        score_static = self.static_block(attn_output)#.mean(dim=1)
        #score_dynamic = self.dynamic_block(attn_output)
        score = self.w_static * score_static# + (1. - self.w_static) * score_dynamic
        return score, attn_output
