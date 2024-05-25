"""
A minimal training script for DiT using Pyth DDP.
"""
import torch as th
# th.backends.cuda.matmul.allow_tf32 = True
# th.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import numpy as np
from time import time   
import os
from diffusers.models import AutoencoderKL
from dataclasses import dataclass
from dit import DiT, DiTConfig,load_pretrained_model
from scheduler import GaussianDiffusion 
th.manual_seed(42)

@dataclass
class TrainConfig:
    image_size: int = 32 # cifar10
    num_epochs: int = 10000
    eval_iters: int = 200
    eval_interval: int = 500
    batch_size: int = 32 
    diffusion_steps = 1000
    n_sampling_steps = 5
    lr: float = 1e-4
    vae_normlizing_const: float = 0.18215 # for stable diffusion vae

    def __post_init__(self):
        assert self.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

train_cfg = TrainConfig()
device = "cuda" if th.cuda.is_available() else "cpu"

# setup diffusion transformer
dit_cfg = DiTConfig(input_size=4,n_heads=4, n_layers=3)
model = DiT(dit_cfg)
model = DiT(dit_cfg)
model.train()  # important! 

# Note that parameter initialization is done within the DiT constructor
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

gd = GaussianDiffusion(train_cfg.diffusion_steps, train_cfg.n_sampling_steps, device=device)


# Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
optimizer = th.optim.AdamW(model.parameters(), lr=train_cfg.lr)

# Setup data:
transform = transforms.Compose([
    transforms.CenterCrop(train_cfg.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

train_dataset  = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
trainloader = th.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size,shuffle=True)

def p_loss(model, x_start, t, y):

    # draw a sample
    noise = th.randn_like(x) #ground truth noise
    a = th.sqrt(gd.alpha_prod)[t].reshape(32,1,1,1)
    b = th.sqrt(1- gd.alpha_prod)[t].reshape(32,1,1,1)

    x_t = a*x_start + b*noise
    B, C = x_t.shape[:2]

    # get predicted noise
    model_output = model(x, t, y)
    noise_pred, model_var_values = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(noise, noise_pred)

for epoch in range(train_cfg.num_epochs):
    for x, y in trainloader:
        x = x.to(device) 
        y = y.to(device)

        with th.no_grad():
            # Map input images to latent space and normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(train_cfg.vae_normlizing_const) 

        # Generate random timesteps for each image latent
        t = th.randint(0, train_cfg.diffusion_steps, (train_cfg.batch_size,), device=device)

        loss = p_loss(model, x, t, y)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        exit()


model.eval() # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


