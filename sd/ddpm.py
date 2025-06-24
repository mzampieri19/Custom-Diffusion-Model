import torch
import numpy as np

class DDPMSampler:
    '''
    DDPM Sampler for Denoising Diffusion Probabilistic Models.
    This class implements the sampling and noise scheduling logic for Denoising Diffusion Probabilistic Models (DDPMs),
    which are generative models used for tasks such as image synthesis. The sampler manages the forward (noising) and
    reverse (denoising) diffusion processes, allowing for both training and inference with customizable noise schedules
    and step counts.
        generator (torch.Generator): PyTorch random number generator for reproducible noise sampling.
        num_training_steps (int, optional): Number of diffusion steps used during training. Default is 1000.
        beta_start (float, optional): Starting value for the noise variance schedule. Default is 0.00085.
        beta_end (float, optional): Ending value for the noise variance schedule. Default is 0.0120.
    Attributes:
        betas (torch.Tensor): Linearly scheduled noise variances for each diffusion step.
        alphas (torch.Tensor): 1 - betas, representing the signal preservation at each step.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas, used for scaling and variance calculations.
        one (torch.Tensor): Scalar tensor with value 1.0, used for boundary conditions.
        generator (torch.Generator): Random number generator for noise sampling.
        num_train_timesteps (int): Number of diffusion steps used during training.
        timesteps (torch.Tensor): Timesteps used for sampling, typically in reversed order for inference.
    Methods:
        set_inference_timesteps(num_inference_steps=50):
            Sets the number of timesteps to use during inference (sampling), adjusting the internal timestep schedule.
        _get_previous_timestep(timestep: int) -> int:
            Returns the previous timestep index for a given current timestep.
        _get_variance(timestep: int) -> torch.Tensor:
            Computes the variance for the current timestep, used for noise sampling during the reverse process.
        set_strength(strength=1):
            Adjusts the starting step for inference based on the given strength, enabling partial denoising (e.g., for image editing).
        step(timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
            Performs a single reverse diffusion step, predicting the sample at the previous timestep given the current latents and model output.
        add_noise(original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
            Adds noise to the original samples at the specified timesteps, used for training or data augmentation.
    Notes:
        - The DDPM sampler enables both full and partial denoising, supporting applications such as image generation and editing.
        - The noise schedule (betas) can be customized for different diffusion behaviors.
        - This module is typically used as part of a larger diffusion model pipeline, where it interacts with a neural network
          that predicts noise at each step.
    Implements a DDPM (Denoising Diffusion Probabilistic Model) sampler for generating images.
    Handles the diffusion process, noise scheduling, and sampling steps.
    '''
    
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Create a linear schedule for betas (noise variance) between beta_start and beta_end
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # Compute alphas (1 - beta) and their cumulative product
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        # Timesteps for training (reversed order)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        """
        Set the number of timesteps to use during inference (sampling).
        Adjusts the timesteps to match the inference steps.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Get the previous timestep index for the current timestep.
        """
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Compute the variance for the current timestep, used for noise sampling.
        """
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength=1):
        """
        Adjust the starting step for inference based on the given strength.
        Used for partial denoising (e.g., image editing).
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Perform a single reverse diffusion step.
        Args:
            timestep: Current timestep index.
            latents: Current latent tensor.
            model_output: Model's predicted noise.
        Returns:
            The predicted sample at the previous timestep.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Estimate the original (denoised) sample
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # Coefficients for combining the original and current samples
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        # Predict the previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        # Add noise except for the last step
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples at the given timesteps.
        Used for training or data augmentation.
        Args:
            original_samples: Clean input samples.
            timesteps: Timesteps at which to add noise.
        Returns:
            Noisy samples.
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples