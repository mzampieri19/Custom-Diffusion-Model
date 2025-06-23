import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    """
    Attention block for spatial feature maps in a VAE decoder.

    This class implements a self-attention mechanism over spatial dimensions of feature maps,
    allowing each spatial location to attend to all others. It is used to enhance the model's
    ability to capture long-range dependencies in image data.

    Args:
        channels (int): Number of channels in the input and output feature maps.

    Attributes:
        groupnorm (nn.GroupNorm): Group normalization layer applied before attention.
        attention (SelfAttention): Self-attention module applied to the normalized input.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, with attention applied.

    Notes:
        - The input is normalized, reshaped for attention, and then reshaped back.
        - A residual connection is added to preserve input information.
    """
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)  # (n, h*w, c)
        x = self.attention(x)
        x = x.transpose(-1, -2)  # (n, c, h*w)
        x = x.view((n, c, h, w))
        x += residue
        return x 

class VAE_ResidualBlock(nn.Module):
    """
    Residual block for convolutional feature processing in a VAE decoder.

    This class implements a two-layer convolutional residual block with group normalization
    and SiLU activation. It supports changing the number of channels between input and output.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        groupnorm_1 (nn.GroupNorm): Group normalization before the first convolution.
        conv_1 (nn.Conv2d): First convolutional layer.
        groupnorm_2 (nn.GroupNorm): Group normalization before the second convolution.
        conv_2 (nn.Conv2d): Second convolutional layer.
        residual_layer (nn.Module): Identity or 1x1 convolution for residual connection.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).

    Notes:
        - Applies normalization, activation, and convolution twice.
        - Adds a residual connection, adjusting channels if needed.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x) 
        x = self.conv_2(x)       
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    """
    Variational Autoencoder (VAE) decoder for image reconstruction.

    This class defines a sequential decoder architecture composed of convolutional, residual,
    attention, normalization, and upsampling layers. It reconstructs images from latent representations.

    Args:
        None

    Attributes:
        Inherits all modules from nn.Sequential, including convolutional, residual, attention,
        normalization, and upsampling layers arranged in a specific order.

    Forward Args:
        x (torch.Tensor): Input latent tensor of shape (batch_size, 4, H, W).

    Returns:
        torch.Tensor: Output image tensor of shape (batch_size, 3, H*8, W*8).

    Notes:
        - The input is scaled before decoding.
        - The architecture progressively upsamples and refines the latent representation.
        - The final output is a reconstructed RGB image.
    """
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
