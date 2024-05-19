
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, H, W, patch_size=16, in_chans=3, embed_dim=100):
        super().__init__()

        self.num_patches = (H * W) // (patch_size ** 2)
        self.patch_size = patch_size

        # since we haveset kernel_size=stride=patch_size, the conv kernel acts on each indivial patch. The conv operation acts as a lienar embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  #(B, C, H, W) -> (B, embed_dim, H, W)
        x = x.flatten(2) #(B, embed_dim, H, W) -> (B, embed_dim, H*W)
        x = x.transpose(1, 2) # (B, embed_dim, H*W) -> (B, H*W, embed_dim)

        return x
