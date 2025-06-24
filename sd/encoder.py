import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    """
    Variational Autoencoder (VAE) Encoder module for image encoding.

    This class implements the encoder part of a VAE, designed to process input images and produce a latent representation
    suitable for generative modeling. The encoder consists of a sequence of convolutional, residual, normalization, and
    attention blocks, progressively downsampling and transforming the input image into a compact latent space.

    The final output is split into mean and log-variance components, from which a latent sample is drawn using the
    reparameterization trick. This enables stochastic sampling while allowing gradients to flow through the network.

    Args:
        None

    Attributes:
        layers (nn.Sequential): Sequence of convolutional, residual, normalization, and attention blocks that process
            the input image tensor. The architecture includes downsampling via strided convolutions, residual connections,
            and an attention block to capture global dependencies.

    Forward Args:
        x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width), representing a batch of RGB images.
        noise (torch.Tensor): Noise tensor of the same shape as the latent representation, used for the reparameterization
            trick to enable stochastic sampling from the latent distribution.

    Returns:
        torch.Tensor: Latent representation tensor of shape (batch_size, latent_channels, latent_height, latent_width),
            sampled from the learned mean and variance, and scaled by a constant factor.

    Notes:
        - The encoder outputs both the mean and log-variance of the latent distribution, which are used to sample the
          latent code via the reparameterization trick.
        - The log-variance is clamped to a fixed range for numerical stability.
        - The output latent representation is scaled by a constant factor (0.18215), which may be used for normalization
          or compatibility with downstream components.
        - This module is typically paired with a corresponding VAE decoder for image generation tasks.
    """
    def __init__(self):
        # Initialize the encoder as a sequence of layers
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # Initial convolution: input RGB image to 128 channels
            VAE_ResidualBlock(128, 128),                  # Residual block: 128 -> 128 channels
            VAE_ResidualBlock(128, 128),                  # Residual block: 128 -> 128 channels
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # Downsample: halve spatial size, keep 128 channels
            
            VAE_ResidualBlock(128, 256),                  # Residual block: 128 -> 256 channels
            VAE_ResidualBlock(256, 256),                  # Residual block: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # Downsample: halve spatial size, keep 256 channels
            
            VAE_ResidualBlock(256, 512),                  # Residual block: 256 -> 512 channels
            VAE_ResidualBlock(512, 512),                  # Residual block: 512 -> 512 channels
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # Downsample: halve spatial size, keep 512 channels
            
            VAE_ResidualBlock(512, 512),                  # Residual block: 512 -> 512 channels
            VAE_ResidualBlock(512, 512),                  # Residual block: 512 -> 512 channels
            VAE_ResidualBlock(512, 512),                  # Residual block: 512 -> 512 channels
            VAE_AttentionBlock(512),                      # Attention block: capture global dependencies
            VAE_ResidualBlock(512, 512),                  # Residual block: 512 -> 512 channels
            
            nn.GroupNorm(32, 512),                        # Group normalization: 32 groups, 512 channels
            nn.SiLU(),                                    # SiLU activation function
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # Reduce channels from 512 to 8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),    # 1x1 convolution: keep 8 channels
        )

    def forward(self, x, noise):
        # Forward pass through all layers in the sequence
        for module in self:
            # If the module is a strided convolution (stride=2), pad input to maintain spatial alignment
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # Split output into mean and log-variance along the channel dimension
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp log-variance for numerical stability
        log_variance = torch.clamp(log_variance, -30, 20)
        # Compute variance and standard deviation
        variance = log_variance.exp()
        stdev = variance.sqrt()
        # Reparameterization trick: sample from latent distribution
        x = mean + stdev * noise
        # Scale latent representation by a constant factor
        x *= 0.18215
        return x
