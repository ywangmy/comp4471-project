import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .MAT import SelfAttention

import math

def get_pretrained(num_classes = 1000, download = False):
    # Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small
    if num_classes == 1000:
        model = torchvision.models.mobilenet_v3_small(weights='DEFAULT', download=download, progress=True)
    else:
        model = torchvision.models.mobilenet_v3_small(weights=None)
    return model

class MatNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_block = nn.Transformer(d_model=None, batch_size=True)
    def forward(self, x):
        """
        Input:
        - x: A feature vector (attention output), shape: (N, F, C, H, W)
        Output:
        - x: Dynamic score
        """
        #N, F, C, H, W = x.shape
        N, D = list(x.size())
        x = torch.linalg.matrix_norm(x) # https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
        x = torch.diff(x, dim=0) # https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff
        x = torch.abs(x)
        x = torch.mean(x, dim=1)
        return x

class staticClassifier_old(nn.Module):
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
        score = F.softmax(relu_output, dim=1)
        score = score[:, 0]
        return score

class staticClassifier(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        # FC as the final classification layer
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.batchnorm1 = nn.BatchNorm1d(in_channels)
        self.PReLU = nn.PReLU()
        self.fc2 = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        """
        Input:
        - x (N, in_channels): A feature vector (attention output)
        Output:
        - score (N, ): Static score
        """

        #print(f'staticClassifier forwarding: alloc {torch.cuda.memory_allocated() / 1024**2}, maxalloc {torch.cuda.max_memory_allocated()  / 1024**2}, reserved {torch.cuda.memory_reserved() / 1024**2}')

        fc1_output = self.fc1(x)
        #batchnorm1_output = self.batchnorm1(fc1_output)
        #prelu_output = self.PReLU(batchnorm1_output)
        prelu_output = self.PReLU(fc1_output)
        fc2_output = self.fc2(prelu_output)
        score = F.softmax(fc2_output, dim=1)
        score = score[:, 0]
        return score

class PositionalEncoding(nn.Module):
    """
    Positional encoding.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        #position = torch.arange(max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        #pe = torch.zeros(max_len, 1, d_model)
        #pe[0, :, 0::2] = torch.sin(position * div_term)
        #pe[0, :, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, dim_embed(d_model)]
        """
        x = x + self.pe[:x.shape[1]]
        return self.dropout(X)

class dynamicClassifier(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, in_channels, out_channels=2, nhead=8, d_hid=1000, nlayers=6, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        #self.pos_encoder = PositionalEncoding(in_channels, dropout)
        #encoder_layers = TransformerEncoderLayer(in_channels, nhead, d_hid, dropout)
        #self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.fc = nn.Linear(in_channels, out_channels)
        self.init_weights()
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, x):
        """
        x: shape [batch_size, seq_len, dim_embed]
        """
        x = self.encoder(x) * math.sqrt(self.in_channels)
        x = self.pos_encoder(x)
        tr_output = self.transformer_encoder(src)
        fc_output = self.fc(tr_output)
        score = F.softmax(fc_output, dim=2)
        score = score[:, 0]
        return score

class ASRID(nn.Module):
    def __init__(self, batch_size, num_frames=1, num_features=1000, num_heads=10, dim_attn=1000, strategy=None):
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

        #self.static_block = staticClassifier_old(in_channels=self.dim_attn)
        self.static_block = staticClassifier(in_channels=self.dim_attn)
        #self.dynamic_block = dynamicClassifier(in_channels=self.dim_attn) # baseline

        # Other parameters
        # self.w_static = torch.rand((1,))
        self.w_static = 1.

    def forward(self, x):
        """
        Input:
        - x (N, F(rames), C(hannels), H, W): Raw information
        Output:
        - score (N, ): Score
        """
        # Change x: (N, F, ...) to x: (N * F, ...)
        N, F, C, H, W = x.size()
        #print(f'x.size()={x.size()}, type={x.type()}')
        #x_imgs = torch.reshape(x, (-1, C, H, W))
        #print(f'main forwarding: alloc {torch.cuda.memory_allocated() / 1024**2}, maxalloc {torch.cuda.max_memory_allocated()  / 1024**2}, reserved {torch.cuda.memory_reserved() / 1024**2}')

        # Feature output (N, F, num_features)
        feat_output = torch.stack([self.efficientNet(img) for img in x])
        #print(f'feat_output.size()={feat_output.size()}')

        # Attention output (N, F, dim_attn)
        # Attention output weight .
        attn_results = [self.multiattn_block(feat) for feat in feat_output]
        attn_output = torch.stack([output for output, _ in attn_results])
        attn_output_weights = torch.stack([output_weight for _, output_weight in attn_results])
        #print(f'attn_output.size()={attn_output.size()}')

        # Static scores (N, F)
        score_static_s = torch.stack([self.static_block(attn) for attn in attn_output])

        # Mean static scores (N,)
        score_static = score_static_s.mean(dim=1)

        # Dynamic scores (N,)
        #score_dynamic = self.dynamic_block(attn_output)

        score = self.w_static * score_static #+ (1. - self.w_static) * score_dynamic
        return score, (attn_output, score_static, score_static) # TODO: change into dynamic
