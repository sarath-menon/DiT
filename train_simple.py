"""
A minimal training script for DiT using Pyth DDP.
"""
import torch as th
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import os
from diffusers.models import AutoencoderKL
from dataclasses import dataclass
from dit import DiT, DiTConfig,load_pretrained_model
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

    def __post_init__(self):
        assert self.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

train_cfg = TrainConfig()
device = "cuda" if th.cuda.is_available() else "cpu"

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model = DiT(dit_cfg)

model = load_pretrained_model(model)
model.train()  # important! 

# Note that parameter initialization is done within the DiT constructor
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
optimizer = th.optim.AdamW(model.parameters(), lr=train_cfg.lr)

# Setup data:
transform = transforms.Compose([
    transforms.CenterCrop(train_cfg.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

train_dataset  = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms)
trainloader = th.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size,shuffle=True)


for epoch in range(train_cfg.num_epochs):
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)

        with th.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        # Generate random timesteps for each image latent
        t = th.randint(0, train_cfg.diffusion_steps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)

        # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        # loss = loss_dict["loss"].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss values:
        # running_loss += loss.item()
        # log_steps += 1
        # train_steps += 1

        # if train_steps % args.log_every == 0:
        #     # Measure training speed:
        #     th.cuda.synchronize()
        #     end_time = time()
        #     steps_per_sec = log_steps / (end_time - start_time)
        #     # Reduce loss history over all processes:
        #     avg_loss = th.tensor(running_loss / log_steps, device=device)
        #     dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        #     avg_loss = avg_loss.item() / dist.get_world_size()
        #     logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
        #     # Reset monitoring variables:
        #     running_loss = 0
        #     log_steps = 0
        #     start_time = time()

        # # Save DiT checkpoint:
        # if train_steps % args.ckpt_every == 0 and train_steps > 0:

model.eval() # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


