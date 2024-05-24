# modified from https://github.com/kjsman/stable-diffusion-pytorch/blob/main/stable_diffusion_pytorch/decoder.py

import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class AttnConfig:
    n_heads: int = 1
    n_embed: int = 1
    dropout: float = 0.0
    device: str = 'cpu'

class AcausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads, self.n_embed = cfg.n_heads, cfg.n_embed

        # key,query,value matrices as a single batch for efficiency
        self.qkv = nn.Linear(cfg.n_embed, 3*cfg.n_embed, bias=False)

        # output layer
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)

        # regularization
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.size()
        head_size = C // self.n_heads

        # get q,k,v, matrices
        q,k,v = self.qkv(x).split(self.n_embed, dim=2)

        # calculate query,key, values for all heads in the batch
        k = k.view(B, T, self.n_heads, head_size ).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, head_size ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, head_size ).transpose(1, 2) # (B, nh, T, hs)

        # Compute dot product between queries and keys for all tokens 
        weights = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1))) # (B,T,hs) @ (B,hs,T) --->  (B,T,T)
 
        weights = F.softmax(weights, dim=-1)
        weights = self.attn_dropout(weights)

        print(weights.shape, v.shape)

        out = weights @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out 

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        # self.attention = SelfAttention(1, channels)
        attn_config = AttnConfig(n_heads=1, n_embed=channels)
        self.attention = AcausalSelfAttention(attn_config)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class StableDiffusionDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x