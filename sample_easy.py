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
model.eval()  # important!


# diffusion = create_diffusion(str(n_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)


# model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# # Sample image:
# samples = diffusion.p_sample_loop(
#     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
# )

def linear_beta_schedule_np(num_diffusion_timesteps):
    scale = 1000  / num_diffusion_timesteps 
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps) 

n_sampling_steps = 5
diffusion_steps=1000
betas = linear_beta_schedule_np(diffusion_steps)



posterior_var = []

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def linear_beta_schedule(diffusion_timesteps):
    scale = 1
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return th.linspace(beta_start, beta_end, diffusion_timesteps) 


import numpy as np

def p_sample(model_output, x, t, T):

    # safety checks
    B, C = x.shape[:2]
    print("B, C: ", B, C)

    # assert t.shape == (B,)
    assert model_output.shape == (B, C * 2, *x.shape[2:])

    model_output, model_var_values = th.split(model_output, C, dim=1)

    betas = linear_beta_schedule(T)
    alphas = 1 - betas
    
    alpha_prod = th.cumprod(alphas, 0)
    alpha_prod_prev = th.cat([th.tensor([1.0]), alpha_prod[:-1]])
    posterior_var =  betas * (1. - alpha_prod_prev) / (1. - alpha_prod)

    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    if len(posterior_var) > 1:
        posterior_var = th.log(th.cat([posterior_var[1].unsqueeze(0), posterior_var[1:]]))
    else:
        posterior_var = th.tensor([])

    # extract values for given timestep
    alpha_prod  = alphas[t]
    alpha_prod_prev= alphas[t-1]
    
    # mean prediction
    # noise_pred = model(x, t, model_kwargs)     
    noise_pred = model_output
    mean_pred =  (x - (betas[t] * noise_pred / th.sqrt(1. - alpha_prod))) * 1 / th.sqrt(alphas[t])

    # var prediction
    # var_fixed = betas[t] * th.eye(x.size(0)) 
    min_log = posterior_var[t]
    max_log = th.log(betas[t])

    # The model_var_values is [-1, 1] for [min_var, max_var].
    frac = (model_var_values + 1) / 2
    model_log_variance = frac * max_log + (1 - frac) * min_log
    var_pred = th.exp(model_log_variance)

    x_prev = mean_pred + var_pred
    return x_prev 

def inference(x,y):

    betas = linear_beta_schedule_np(diffusion_steps)

    spaced_diffusion = SpacedDiffusion(
    use_timesteps=space_timesteps(diffusion_steps,[n_sampling_steps]),
    betas=betas,
    model_mean_type=gd.ModelMeanType.EPSILON,
    model_var_type=gd.ModelVarType.LEARNED_RANGE,
    loss_type = gd.LossType.MSE)

    # time indices in reverse
    # indices = list(range(1, n_sampling_steps + 1))[::-1]
    indices = list(range(spaced_diffusion.num_timesteps))[::-1]

    for i in indices:
        t = th.tensor([i] * x.shape[0], device="cpu") 

        model_output = model.forward_with_cfg(x, t, y, cfg_scale)

        x = p_sample(model_output, x, i, spaced_diffusion.num_timesteps) 
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

# print("z: ", z[0,0,:10])

samples = inference(z,y)

# convert image latent to image
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
path = os.getcwd()
output_dir = os.path.join(path, "out")
save_image(samples, os.path.join(output_dir, "sample.png"), nrow=1, normalize=True, value_range=(-1, 1))

