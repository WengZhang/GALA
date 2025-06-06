import numpy as np
import torch


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_std=1.0):
    noise = torch.randn_like(x_0, device=x_0.device) * noise_std
    alphas_t_sqrt = alphas_bar_sqrt[t].view(-1, 1)
    one_minus_alphas_bar_t_sqrt = one_minus_alphas_bar_sqrt[t].view(-1, 1)
    x_t = alphas_t_sqrt * x_0 + one_minus_alphas_bar_t_sqrt * noise
    return x_t

class Diffusion(torch.nn.Module):
    def __init__(self, beta_schedule='linear', beta_start=1e-4, beta_end=2e-2, t_min=10, t_max=1000, noise_std=0.05):
        super().__init__()
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start  
        self.beta_end = beta_end   
        self.t_min = t_min   
        self.t_max = t_max
        self.noise_std = float(noise_std)  
        self.num_timesteps = None  
        self.update_T()  

    def set_diffusion_process(self, t, beta_schedule):  
        t = int(t)  
        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=t,  
        )
        betas = self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = self.alphas = 1.0 - betas
        alphas_cumprod = torch.cat([torch.tensor([1.]), alphas.cumprod(dim=0)])
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

    def update_T(self):
        self.set_diffusion_process(self.t_max, self.beta_schedule)

    def forward(self, x_0):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=device)   
        alphas_bar_sqrt = self.alphas_bar_sqrt.to(device)  
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt.to(device)  
        x_t = q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise_std=self.noise_std)
        return x_t, t.view(-1, 1)

#----------------------------------------------------------------------------
