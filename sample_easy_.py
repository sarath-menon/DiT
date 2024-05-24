import torch as th
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from download import find_model
from dit import DiT, DiTConfig, load_pretrained_model
# from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import os
import numpy as np
from diffusion.respace import  space_timesteps
from diffusion.respace import SpacedDiffusion, space_timesteps
import diffusion.gaussian_diffusion as gd
from tqdm.auto import tqdm
import numpy as np

th.manual_seed(1)
device = "cuda" if th.cuda.is_available() else "cpu"

diffusion_steps = 1000
n_sampling_steps = 5
cfg_scale = 4.0
class_labels = [11] # Labels to condition the model with (feel free to change):

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model = load_pretrained_model(model)
model.eval()  

# diffusion = create_diffusion(str(n_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

def linear_beta_schedule_np(num_diffusion_timesteps):
    scale = 1000.0  / num_diffusion_timesteps 
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps) 

def linear_beta_schedule(diffusion_timesteps):
    scale = 1
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return th.linspace(beta_start, beta_end, diffusion_timesteps) 

betas = linear_beta_schedule_np(diffusion_steps)
spaced_diffusion = SpacedDiffusion(
use_timesteps=space_timesteps(diffusion_steps,[n_sampling_steps]),
betas=betas,
model_mean_type=gd.ModelMeanType.EPSILON,
model_var_type=gd.ModelVarType.LEARNED_RANGE,
loss_type = gd.LossType.MSE)

betas = th.from_numpy(spaced_diffusion.betas).float().to(device)
alphas = 1. - betas
alpha_prod = th.cumprod(alphas, 0)
alpha_prod_prev = th.cat([th.tensor([1.0]), alpha_prod[:-1]])
posterior_var =  betas * (1. - alpha_prod_prev) / (1. - alpha_prod)

# log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
if len(posterior_var) > 1:
    posterior_var = th.log(th.cat([posterior_var[1].unsqueeze(0), posterior_var[1:]]))
else:
    posterior_var = th.tensor([])

 
@th.no_grad()
def p_sample_loop(model_output, x, t):
    # safety checks
    B, C = x.shape[:2]
    assert model_output.shape == (B, C * 2, *x.shape[2:])

    # get model output
    noise_pred, model_var_values = th.split(model_output, C, dim=1)
   
    # mean prediction
    coeff1 = th.sqrt(1./alphas) 
    coeff2 = betas * th.sqrt(1./alphas) * th.sqrt(1./(1-alpha_prod))
    mean_pred = coeff1[t] * x - coeff2[t] * noise_pred

    # var prediction
    min_log = th.full_like(x, posterior_var[t])
    max_log = th.full_like(x, th.log(betas[t]))
    frac = (model_var_values + 1) / 2
    model_log_var = frac * max_log + (1 - frac) * min_log
    std_dev_pred = th.exp(0.5 * model_log_var)

    # inference
    nonzero_mask = 0. if t == 0 else 1.; 
    noise = th.randn_like(x)
    x_prev = mean_pred + nonzero_mask * std_dev_pred * noise

    return x_prev 

def inference(x,y):

    # time indices in reverse
    indices = list(range(spaced_diffusion.num_timesteps))[::-1]
    indices = tqdm(indices)
    map_ts = th.tensor([  0, 250, 500, 749, 999])

    for i in indices:
        t = th.tensor([map_ts[i]] * x.shape[0], device="cpu") 
        model_output = model.forward_with_cfg(x, t, y, cfg_scale)

        x = p_sample_loop(model_output, x,  i) 
    return x

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

