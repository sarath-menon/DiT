import torch as th
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from download import find_model
from dit import DiT, DiTConfig, load_pretrained_model
from diffusers.models import AutoencoderKL
import os
import numpy as np
from tqdm.auto import tqdm
import numpy as np
from dataclasses import dataclass

th.manual_seed(1)
device = "cuda" if th.cuda.is_available() else "cpu"

diffusion_steps = 1000
n_sampling_steps = 5
cfg_scale = 4.0
class_labels = [11] # Labels to condition the model with (feel free to change)

def linear_beta_schedule(diffusion_timesteps):
    scale = 1
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return th.linspace(beta_start, beta_end, diffusion_timesteps) 

@dataclass
class GaussianDiffusionParams:
    betas: th.Tensor
    device: str
    def __post_init__(self):
        self.betas = self.betas.float().to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_prod = th.cumprod(self.alphas, 0)
        self.alpha_prod_prev = th.cat([th.tensor([1.0], device=self.device), self.alpha_prod[:-1]])
        self.posterior_var = self.betas * (1. - self.alpha_prod_prev) / (1. - self.alpha_prod)
        if len(self.posterior_var) > 1:
            self.posterior_var = th.log(th.cat([self.posterior_var[1].unsqueeze(0), self.posterior_var[1:]]))
        else:
            self.posterior_var = th.tensor([], device=self.device)

def space_timesteps(num_timesteps, section_counts):
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        stride = (size - 1) / (section_count - 1) if section_count > 1 else 0
        steps = [start_idx + round(stride * j) for j in range(section_count)]
        all_steps.extend(steps)
        start_idx += size
    return all_steps

def respace_betas(betas, use_timesteps):
    last_alpha_prod = 1.0
    alphas = 1. - betas
    alpha_prod = th.cumprod(alphas, 0)
    new_betas = []
    timestep_map = []
    for i, alpha_prod in enumerate(alpha_prod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_prod / last_alpha_prod)
            last_alpha_prod = alpha_prod
            timestep_map.append(i)
    return th.tensor(new_betas), timestep_map

@th.no_grad()
def p_sample_loop(model_output, x, t, gd):
    # safety checks
    B, C = x.shape[:2]
    assert model_output.shape == (B, C * 2, *x.shape[2:])

    # get model output
    noise_pred, model_var_values = th.split(model_output, C, dim=1)
   
    # mean prediction
    coeff1 = th.sqrt(1./gd.alphas) 
    coeff2 = gd.betas * th.sqrt(1./gd.alphas) * th.sqrt(1./(1-gd.alpha_prod))
    mean_pred = coeff1[t] * x - coeff2[t] * noise_pred

    # var prediction
    min_log = th.full_like(x, gd.posterior_var[t])
    max_log = th.full_like(x, th.log(gd.betas[t]))
    frac = (model_var_values + 1) / 2
    model_log_var = frac * max_log + (1 - frac) * min_log
    std_dev_pred = th.exp(0.5 * model_log_var)

    # inference
    nonzero_mask = 0. if t == 0 else 1.; 
    noise = th.randn_like(x)
    x_prev = mean_pred + nonzero_mask * std_dev_pred * noise

    return x_prev 

def inference(x,y):

    # create params for gaussian diffusion
    indices = list(range(n_sampling_steps))[::-1]
    indices = tqdm(indices) # for progres bar
    map_ts = space_timesteps(diffusion_steps, [n_sampling_steps])

    betas = linear_beta_schedule(diffusion_steps)
    betas, _ = respace_betas(betas, map_ts)
    gd = GaussianDiffusionParams(betas, device)

    for i in indices:
        t = th.tensor([map_ts[i]] * x.shape[0], device="cpu") 
        model_output = model.forward_with_cfg(x, t, y, cfg_scale)

        x = p_sample_loop(model_output, x,  i, gd) 
    return x

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model = load_pretrained_model(model)
model.eval()  

# diffusion = create_diffusion(str(n_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

 # Convert image class to noise latent:
latent_size = dit_cfg.input_size
n = len(class_labels)
z = th.randn(n, 4, latent_size, latent_size, device=device)
y = th.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = th.cat([z, z], 0)
y_null = th.tensor([1000] * n, device=device)
y = th.cat([y, y_null], 0)

samples = inference(z,y)

# convert image latent to image
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
path = os.getcwd()
output_dir = os.path.join(path, "out")
save_image(samples, os.path.join(output_dir, "sample.png"), nrow=1, normalize=True, value_range=(-1, 1))

