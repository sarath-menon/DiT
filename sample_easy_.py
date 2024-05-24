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
n_sampling_steps = 10
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

n_sampling_steps = 5
diffusion_steps=1000

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

@th.no_grad()
def p_sample_loop(model_output, x, t, T, betas):

    betas = th.from_numpy(betas).float().to(device)
    # safety checks
    B, C = x.shape[:2]

    # assert t.shape == (B,)
    assert model_output.shape == (B, C * 2, *x.shape[2:])
    model_output, model_var_values = th.split(model_output, C, dim=1)
    print("model_output: ", model_output[0,0,0,:2])
    # print("model_var_values: ", model_var_values[0,0,0,:2])

    # betas = linear_beta_schedule(T)
    alphas = 1. - betas
    
    alpha_prod = th.cumprod(alphas, 0)
    alpha_prod_prev = th.cat([th.tensor([1.0]), alpha_prod[:-1]])
    posterior_var =  betas * (1. - alpha_prod_prev) / (1. - alpha_prod)

   
   
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    if len(posterior_var) > 1:
        posterior_var = th.log(th.cat([posterior_var[1].unsqueeze(0), posterior_var[1:]]))
    else:
        posterior_var = th.tensor([])
    
    # mean prediction  
    noise_pred = model_output

    # Implementation by : Ho, OpenAI, Meta (Working)
    # a = th.sqrt(1. / alpha_prod)
    # b = th.sqrt(1. / alpha_prod - 1)
    # x_start_pred = (a[t]* x) - (b[t]* noise_pred)
    # coeff1 = th.sqrt(alpha_prod_prev) * betas  / (1 - alpha_prod)
    # coeff2 = th.sqrt(alphas) * (1 - alpha_prod_prev)  / (1 - alpha_prod) 
    # mean_pred = coeff1[t] * x_start_pred + coeff2[t] * x

    # Eqn in paper (working)
    coeff1 = th.sqrt(1./alphas) 
    coeff2 = betas * th.sqrt(1./alphas) * th.sqrt(1./(1-alpha_prod))
    mean_pred = coeff1[t] * x - coeff2[t] * noise_pred

    # var prediction
    min_log = posterior_var[t]
    max_log = th.log(betas[t])

    min_log = th.full_like(x, min_log)
    max_log = th.full_like(x, max_log)

    # The model_var_values is [-1, 1] for [min_var, max_var].
    frac = (model_var_values + 1) / 2
    model_log_variance = frac * max_log + (1 - frac) * min_log
    std_dev_pred = th.exp(0.5 * model_log_variance)

    noise = th.randn_like(x)
    nonzero_mask = 1.
    if t==0:
        nonzero_mask = 0.

    x_prev = mean_pred + nonzero_mask * std_dev_pred * noise

    # print("x: ", x[0,0,0,:2])
    # print("x_prev: ", x_prev[0,0,0,:2])


    print("mean_pred: ", mean_pred[0,0,0,:2])
    print("log_variance: ", model_log_variance[0,0,0,:2])

    # print("min_log: ", min_log[0,0,0,0])
    # print("max_log: ", max_log[0,0,0,0])
    # print("betas: ", betas)

    # exit()

    # x_prev = mean_pred + var_pred
    return x_prev 

def inference(z,y):

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

    map_ts = th.tensor([  0, 250, 500, 749, 999])

    x = z

    for i in indices:
        print(i)
        t = th.tensor([map_ts[i]] * x.shape[0], device="cpu") 

        model_output = model.forward_with_cfg(x, t, y, cfg_scale)

        x = p_sample_loop(model_output, x,  i, spaced_diffusion.num_timesteps, spaced_diffusion.betas) 
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

