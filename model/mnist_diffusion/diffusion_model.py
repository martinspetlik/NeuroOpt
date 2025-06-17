import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from model.auxiliary_functions import extract


class DiffusionModel(nn.Module):
    def __init__(self, cnn_model, noise_scheduler):
        """
        DiffusionModel encapsulates the forward (noise addition) and reverse (denoising) diffusion processes.
        :param cnn_model: a neural network used to predict noise
        :param noise_scheduler: an object that holds diffusion schedule parameters
        """
        super(DiffusionModel, self).__init__()
        self._name = "DiffusionModel"
        self.model = cnn_model  # CNN to predict noise at each timestep
        self.noise_scheduler = noise_scheduler  # Contains precomputed diffusion coefficients

    def q_sample(self, x_start, t, noise):
        """
        Forward diffusion process: add noise to the original data x_start at timestep t.
        :param x_start: original clean input
        :param t: current timestep (per sample)
        :param noise: sampled Gaussian noise
        :return: noised data at timestep t
        """
        # Extract diffusion coefficients for timestep t
        sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Apply forward diffusion equation
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, samples, noise=None):
        """
        Forward pass for training: simulate a forward diffusion step and predict the noise.
        :param samples: tuple (images, conds) where conds are optional conditioning inputs
        :param noise: externally provided noise, or sampled if None
        :return: actual noise and model-predicted noise
        """
        batch_size = samples[0].shape[0]
        images, conds = samples

        if noise is None:
            noise = torch.randn_like(images)  # Sample Gaussian noise

        # Randomly sample diffusion timesteps per sample in the batch
        timestamp = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=images.device).long()

        # Add noise to clean images
        x_noised = self.q_sample(images, timestamp, noise=noise)

        # Predict the noise from the noised image and timestamp
        predicted_noise = self.model((x_noised, conds), timestamp)

        return noise, predicted_noise

    def p_samples(self, x, timestamp, labels=None):
        """
        Reverse diffusion step: denoise data x at timestep t.
        :param x: current noisy sample
        :param timestamp: timestep t
        :param labels: optional conditioning labels
        :return: denoised sample at timestep t-1
        """
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full((b,), timestamp, device=device, dtype=torch.long)

        # Ensure necessary scheduler components are on the correct device (GPU)
        if torch.cuda.is_available():
            batched_timestamps = batched_timestamps.cuda()
            self.noise_scheduler.betas = self.noise_scheduler.betas.cuda()
            self.noise_scheduler.sqrt_recip_alphas = self.noise_scheduler.sqrt_recip_alphas.cuda()
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod = self.noise_scheduler.sqrt_one_minus_alphas_cumprod.cuda()
            self.noise_scheduler.posterior_variance = self.noise_scheduler.posterior_variance.cuda()

        data = x.float()

        # Predict noise using the model
        preds = self.model((data, labels), batched_timestamps)

        # Extract coefficients for timestep t
        betas_t = extract(self.noise_scheduler.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(self.noise_scheduler.sqrt_recip_alphas, batched_timestamps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape)

        # Compute the predicted mean (denoised version)
        predicted_mean = sqrt_recip_alphas_t * (data - betas_t * preds / sqrt_one_minus_alphas_cumprod_t)

        if timestamp == 0:
            # Final timestep: return the predicted mean (no noise added)
            return predicted_mean
        else:
            # Add noise sampled from the posterior variance
            posterior_variance = extract(self.noise_scheduler.posterior_variance, batched_timestamps, x.shape)
            noise = torch.randn_like(x)
            scaled_noise = torch.sqrt(posterior_variance) * noise
            return predicted_mean + scaled_noise

    @torch.inference_mode()
    def sample(self, batch_size, shape, labels=None, inverse_transform=None):
        """
        Generate samples by reversing the diffusion process starting from pure noise.
        :param batch_size: number of samples to generate
        :param shape: shape of a single sample (e.g., (channels, height, width))
        :param labels: optional conditioning labels
        :param inverse_transform: optional function to transform samples back to original domain
        :return: (inverse-transformed samples, raw model output samples)
        """
        shape = (batch_size, 1, *shape)  # Final tensor shape: (B, C, H, W, ...)
        samples = torch.randn(shape)  # Start from Gaussian noise

        if torch.cuda.is_available():
            samples = samples.cuda()
            if labels is not None:
                labels = labels.cuda()

        # Fallback to random labels for unconditional generation
        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=samples.device)

        # Iterate over the reverse diffusion process
        for t in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)),
                      total=self.noise_scheduler.num_gen_timesteps):
            samples = self.p_samples(samples, t, labels=labels)

        inv_samples = samples
        if inverse_transform is not None:
            # Apply inverse transformation (e.g., denormalization)
            inv_samples = inverse_transform(np.squeeze(samples, axis=0))

        return inv_samples, samples
