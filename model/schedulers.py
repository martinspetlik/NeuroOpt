import torch
import torch.nn as nn
from torch.functional import F


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    # Linearly increases beta from beta_start to beta_end over `timesteps`.
    # Beta controls the variance of Gaussian noise at each diffusion step.
    # Common choice in early diffusion models (e.g., DDPM).

    scale = 1000 / timesteps
    beta_start = scale * 0.0001      # Starting small noise
    beta_end = scale * 0.02          # Ending with larger noise
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    # Returns: βₜ linearly spaced between β_start and β_end.


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    # Cosine schedule for beta derived from cumulative product of alphas.
    # Proposed in Improved DDPM (Nichol & Dhariwal, 2021).
    # Alphas_cumprod ~ cos²((t/T + s)/(1+s) * π/2)

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to 1 at t=0

    # Derive βₜ from ᾱₜ: βₜ = 1 - ᾱₜ / ᾱₜ₋₁
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = 3, end: int = 3, tau: int = 1
) -> torch.Tensor:
    # Uses a sigmoid function to control how ᾱₜ (and thus βₜ) evolve over time.
    # ᾱₜ is derived from a scaled sigmoid: ᾱₜ = normalized sigmoid(-((t * (end-start) + start)/tau))

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps

    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()

    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to 1 at t=0

    # βₜ = 1 - ᾱₜ / ᾱₜ₋₁
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class NoiseScheduler(nn.Module):
    # Wrapper class for selecting and managing beta schedules.
    # Computes key values for the forward diffusion process.

    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(self, beta_scheduler_type, num_timesteps, num_gen_timesteps=None, scheduler_kwargs=None, use_cuda=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_gen_timesteps = num_gen_timesteps if num_gen_timesteps is not None else num_timesteps
        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler_type)
        if self.beta_scheduler_fn is None:
            raise ValueError("An unknown beta scheduler type: {}".format(beta_scheduler_type))

        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        # βₜ: noise schedule
        self.betas = self.beta_scheduler_fn(num_timesteps, **scheduler_kwargs)

        # αₜ = 1 - βₜ
        self.alphas = 1.0 - self.betas

        # ᾱₜ = ∏ₛ=1^t αₛ
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱₜ₋₁ with ᾱ₀ = 1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance (used in reverse process):
        # posterior_varₜ = βₜ * (1 - ᾱₜ₋₁) / (1 - ᾱₜ)
        self.posterior_variance = (
                self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # Precomputed terms for sampling:
        # √(1 / αₜ)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # √(ᾱₜ), used for noise prediction
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # √(1 - ᾱₜ), used for adding noise
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Move to GPU if available
        if torch.cuda.is_available() and use_cuda:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cuda()
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.cuda()
