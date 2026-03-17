import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.ddpm.noise_schedule import get_beta_schedule


class GaussianDiffusion:
    """
    Implements the forward and reverse processes of a Denoising Diffusion
    Probabilistic Model (DDPM).

    Forward (q):  progressively adds Gaussian noise to data.
    Reverse (p):  iteratively denoises starting from pure Gaussian noise.
    """

    def __init__(self, timesteps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        self.device = device

        betas = get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        self.betas = betas.to(device)

        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = 1.0 / torch.sqrt(alphas).to(device)

        # Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """Forward process: sample x_t given x_0 and timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def training_loss(self, model, x_0):
        """
        Compute the DDPM training loss: MSE between true noise and predicted noise.

        1. Sample random timesteps
        2. Add noise to x_0 via the forward process
        3. Predict the noise with the model
        4. Return MSE loss
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        """Single reverse-process step: x_t -> x_{t-1}."""
        batch_size = x_t.size(0)
        t_batch = torch.full(
            (batch_size,), t_index, device=x_t.device, dtype=torch.long
        )

        predicted_noise = model(x_t, t_batch)

        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_index]

        mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise
        )

        if t_index == 0:
            return mean

        variance = self.posterior_variance[t_index]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, model, shape, progress=True):
        """
        Full reverse process: generate images starting from Gaussian noise.

        x_T -> x_{T-1} -> ... -> x_0
        """
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)

        steps = reversed(range(self.T))
        if progress:
            steps = tqdm(steps, total=self.T, desc="Sampling")

        for t in steps:
            x = self.p_sample(model, x, t)

        return x
