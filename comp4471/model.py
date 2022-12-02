import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from .MAT import MultiHeadAttention
from .MAT import SelfAttention

def get_pretrained(num_classes = 1000):
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    if num_classes == 1000:
        # model = torchvision.models.resnet18(weights='DEFAULT', progress=True)
        model = torchvision.models.mobilenet_v3_small(weights='DEFAULT', progress=True)
    else:
        model = torchvision.models.mobilenet_v3_small(weights=None)
    return model

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
        self.fc1 = nn.Linear(in_channels, in_channels//2)
        self.batchnorm1 = nn.BatchNorm1d(in_channels//2)
        self.PReLU = nn.PReLU()
        self.fc2 = nn.Linear(in_channels//2, out_channels)
    def forward(self, x):
        """
        Input:
        - x (N, in_channels): A feature vector (attention output)
        Output:
        - score (N, out_channels=2): Static score
        """

        #print(f'staticClassifier forwarding: alloc {torch.cuda.memory_allocated() / 1024**2}, maxalloc {torch.cuda.max_memory_allocated()  / 1024**2}, reserved {torch.cuda.memory_reserved() / 1024**2}')

        fc1_output = self.fc1(x)
        batchnorm1_output = self.batchnorm1(fc1_output)
        prelu_output = self.PReLU(batchnorm1_output)
        fc2_output = self.fc2(prelu_output)
        score = F.softmax(fc2_output, dim=2)
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

        #print(f'main forwarding: alloc {torch.cuda.memory_allocated() / 1024**2}, maxalloc {torch.cuda.max_memory_allocated()  / 1024**2}, reserved {torch.cuda.memory_reserved() / 1024**2}')

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
