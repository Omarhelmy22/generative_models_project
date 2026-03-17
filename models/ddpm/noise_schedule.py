import torch
import math


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in 'Improved DDPM'."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def get_beta_schedule(schedule_name, timesteps, beta_start=1e-4, beta_end=0.02):
    if schedule_name == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")
