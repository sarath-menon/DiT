# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from dit import DiT, DiTConfig
import argparse
import os

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8

    print("Loading model")

    # # Load model:
    #
    # model = DiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes
    # ).to(device)

    dit_config = DiTConfig()
    model = DiT(dit_config)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)

    # model.load_state_dict(state_dict)

     # Load state dict layer by layer:
    own_state = model.state_dict()
    
    # Define a mapping from old keys to new keys
    key_mapping = {}
    for key in state_dict.keys():
        if ".adaLN_modulation.1.weight" in key:
            new_key = key.replace(".adaLN_modulation.1.weight", ".adaLN_modulation.linear.weight")
            key_mapping[key] = new_key
        elif ".adaLN_modulation.1.bias" in key:
            new_key = key.replace(".adaLN_modulation.1.bias", ".adaLN_modulation.linear.bias")
            key_mapping[key] = new_key
        elif "y_embedder" in key:
            new_key = key.replace("y_embedder", "label_embedder")
            key_mapping[key] = new_key
        elif "x_embedder" in key:
            new_key = key.replace("x_embedder", "patch_embedder")
            key_mapping[key] = new_key
        elif "t_embedder" in key:
            new_key = key.replace("t_embedder", "timestep_embedder")
            key_mapping[key] = new_key

    for name, param in state_dict.items():
        # Map the state_dict name to the model's state_dict name if necessary
        mapped_name = key_mapping.get(name, name)
        if mapped_name in own_state:
            try:
                own_state[mapped_name].copy_(param)
                # print(f"Layer {mapped_name} loaded successfully")
            except Exception as e:
                print(f"Failed to load {mapped_name}. Reason: {e}")
        else:
            print(f"{mapped_name} not found in the model's state_dict")

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    print("Model loaded")

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = [11]

    # Convert image class to noise latent:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
