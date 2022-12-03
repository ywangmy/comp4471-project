import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

#class ScaledDotProduct(nn.Module):
#    def __init__(self,input_size):
#        super(ScaledDotProduct,self).__init__()
#        self.sqrt_dim=np.sqrt(input_size)
#    def forward(self, query, key, value, mask=None, dropout=None):
#        scores=torch.matmul(query,key.transpose(-2,-1))/self.sqrt_dim
#        if mask!=None: scores.masked_fill_(mask.view(scores.size()), -float('Inf'))
#        if dropout!=None: attn_output_weights=dropout(attn_output_weights)
#        attn_output_weights=F.softmax(scores,dim=-1)
#        attn_output=torch.matmul(attn_output_weights,value)
#        return attn_output, attn_output_weights

#class MultiHeadAttention(nn.Module):
#    def __init__(self,num_heads,hidden_size,dropout=0.1):
#        super(MultiHeadAttention,self).__init__()
#        assert int (hidden_size/num_heads)
#        self.num_heads = num_heads
#        self.hidden_size = hidden_size
#        self.input_size = int(hidden_size/num_heads)
#        self.scaled_dot_attn = ScaledDotProduct(self.input_size)

#    def forward(self, query, key, value, mask=None):
#        if mask!=None: mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
#        num_batches=query.size(0)
#        query, key, value = [l(x).view(num_batches, -1, self.num_heads, self.input_size).transpose(1, 2)
#             for l, x in zip(self.linears, (query, key, value))]
#        x, self.attn=self.scaled_dot_attn(query,value)
#        n=self.num_heads*self.input_size
#        x=x.transpos(1,2).contiguous().view(num_batches,-1,n)
#        return self.linears[-1](x)

class SelfAttention(nn.Module):

    # (N, F, num_features, num_heads)
    def __init__(self, batch_size, num_frames, num_features, num_heads, embed_dim, **kwargs):
        super(SelfAttention, self).__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_block = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.fc_query = nn.Linear(self.num_features, self.attn_block.embed_dim, bias=False)
        self.fc_key = nn.Linear(self.num_features, self.attn_block.kdim, bias=False)
        self.fc_value = nn.Linear(self.num_features, self.attn_block.vdim, bias=False)

        self.relu = F.relu

    def forward(self, x):
        """
        Input:
        - x (N, D): feature output
        Outpur:
        - attn_output: Attention output
        - attn_output_weights: Attention weights
        """
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)
        attn_output, attn_output_weights = self.attn_block(query, key, value)
        attn_output = self.relu(attn_output)
        return attn_output, attn_output_weights

