import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from download import find_model
from dit import DiT, DiTConfig, load_pretrained_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import os

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

n_sampling_steps = 5
cfg_scale = 4.0

dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model = load_pretrained_model(model)

model.eval()  # important!
diffusion = create_diffusion(str(n_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

# Labels to condition the model with (feel free to change):
class_labels = [11]

# Convert image class to noise latent:
latent_size = dit_cfg.input_size
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
)

samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

# convert image latent to image
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
# Inside the main function, before save_image call
path = os.getcwd()
output_dir = os.path.join(path, "out")

# Now you can safely save the image
save_image(samples, os.path.join(output_dir, "sample.png"), nrow=1, normalize=True, value_range=(-1, 1))

