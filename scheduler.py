
import torch as th
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

class GaussianDiffusion:
    def __init__(self, diffusion_steps, n_sampling_steps, device='cpu'):
        self.diffusion_steps = diffusion_steps
        self.n_sampling_steps = n_sampling_steps
        self.device = device

        self.betas = self.linear_beta_schedule(diffusion_steps).float().to(device)
        self.map_ts = self.space_timesteps(diffusion_steps, [n_sampling_steps])
        self.betas, _ = self.respace_betas(self.betas, self.map_ts)
        
        self.alphas = 1. - self.betas
        self.alpha_prod = th.cumprod(self.alphas, 0)
        self.alpha_prod_prev = th.cat([th.tensor([1.0], device=self.device), self.alpha_prod[:-1]])
        self.posterior_var = self.betas * (1. - self.alpha_prod_prev) / (1. - self.alpha_prod)
        if len(self.posterior_var) > 1:
            self.posterior_var = th.log(th.cat([self.posterior_var[1].unsqueeze(0), self.posterior_var[1:]]))
        else:
            self.posterior_var = th.tensor([], device=self.device)
    
    def linear_beta_schedule(self, diffusion_timesteps):
        scale = 1
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return th.linspace(beta_start, beta_end, diffusion_timesteps) 


    def space_timesteps(self, num_timesteps, section_counts):
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

    def respace_betas(self, betas, use_timesteps):
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

