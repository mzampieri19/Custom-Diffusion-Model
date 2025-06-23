import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    """
    Time step embedding module for diffusion models.

    This class embeds the scalar time step into a higher-dimensional space, enabling the network to condition its
    computations on the current diffusion step. The embedding is processed through two linear layers with a SiLU
    activation in between, expanding the representation and allowing for richer conditioning.

    Args:
        n_embd (int): Dimensionality of the time embedding.

    Attributes:
        linear_1 (nn.Linear): First linear layer that expands the input embedding.
        linear_2 (nn.Linear): Second linear layer that further processes the expanded embedding.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, n_embd), representing the time step embeddings.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 4 * n_embd), representing the processed time embeddings.

    Notes:
        - The time embedding is typically generated from a sinusoidal or learned embedding of the time step.
        - This module is used to condition the UNet at various layers, enabling time-dependent behavior.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # Expands the embedding dimension
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)  # Further processes the embedding

    def forward(self, x):
        x = self.linear_1(x)  # Linear projection
        x = F.silu(x)         # SiLU activation
        x = self.linear_2(x)  # Second linear projection
        return x

class UNET_ResidualBlock(nn.Module):
    """
    Residual block for UNet with time conditioning.

    This block applies normalization, activation, and convolution to the input features, and conditions the output
    on a time embedding. It includes a residual connection, optionally using a 1x1 convolution if the input and output
    channels differ.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        n_time (int, optional): Dimensionality of the time embedding. Default is 1280.

    Attributes:
        groupnorm_feature (nn.GroupNorm): Group normalization for input features.
        conv_feature (nn.Conv2d): Convolution for feature transformation.
        linear_time (nn.Linear): Linear layer to project time embedding to feature channels.
        groupnorm_merged (nn.GroupNorm): Group normalization after merging time and features.
        conv_merged (nn.Conv2d): Convolution for merged features.
        residual_layer (nn.Module): Identity or 1x1 convolution for residual connection.

    Forward Args:
        feature (torch.Tensor): Input feature map of shape (batch_size, in_channels, H, W).
        time (torch.Tensor): Time embedding tensor of shape (batch_size, n_time).

    Returns:
        torch.Tensor: Output feature map of shape (batch_size, out_channels, H, W).

    Notes:
        - The time embedding is broadcast and added to the feature map after the first convolution.
        - Residual connections help stabilize training and enable deeper networks.
    """
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)  # Normalize input features
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Feature conv
        self.linear_time = nn.Linear(n_time, out_channels)  # Project time embedding to feature channels
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)  # Normalize merged features
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # Merge conv
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()  # Use identity if channels match
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)  # 1x1 conv for skip
    
    def forward(self, feature, time):
        residue = feature  # Save input for skip connection
        feature = self.groupnorm_feature(feature)  # Normalize
        feature = F.silu(feature)  # Activation
        feature = self.conv_feature(feature)  # Convolution
        time = F.silu(time)  # Activate time embedding
        time = self.linear_time(time)  # Project time embedding
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)  # Add time embedding (broadcast)
        merged = self.groupnorm_merged(merged)  # Normalize merged
        merged = F.silu(merged)  # Activation
        merged = self.conv_merged(merged)  # Convolution
        return merged + self.residual_layer(residue)  # Add skip connection

class UNET_AttentionBlock(nn.Module):
    """
    Attention block for UNet with self-attention, cross-attention, and GEGLU feed-forward.

    This block applies group normalization and a 1x1 convolution, followed by multi-head self-attention, cross-attention
    with an external context, and a gated feed-forward network (GEGLU). Residual connections are used throughout.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Dimensionality per attention head.
        d_context (int, optional): Dimensionality of the context for cross-attention. Default is 768.

    Attributes:
        groupnorm (nn.GroupNorm): Group normalization for input.
        conv_input (nn.Conv2d): 1x1 convolution for input projection.
        layernorm_1 (nn.LayerNorm): Layer normalization before self-attention.
        attention_1 (SelfAttention): Multi-head self-attention module.
        layernorm_2 (nn.LayerNorm): Layer normalization before cross-attention.
        attention_2 (CrossAttention): Multi-head cross-attention module.
        layernorm_3 (nn.LayerNorm): Layer normalization before feed-forward.
        linear_geglu_1 (nn.Linear): Linear layer for GEGLU feed-forward (value and gate).
        linear_geglu_2 (nn.Linear): Linear layer for GEGLU output projection.
        conv_output (nn.Conv2d): 1x1 convolution for output projection.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, H, W).
        context (torch.Tensor): Context tensor for cross-attention.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels, H, W).

    Notes:
        - Self-attention enables spatial positions to interact.
        - Cross-attention allows conditioning on external information (e.g., text embeddings).
        - GEGLU feed-forward provides non-linear transformation with gating.
    """
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)  # Normalize input
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)  # 1x1 conv
        self.layernorm_1 = nn.LayerNorm(channels)  # LayerNorm before self-attention
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)  # Self-attention
        self.layernorm_2 = nn.LayerNorm(channels)  # LayerNorm before cross-attention
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)  # Cross-attention
        self.layernorm_3 = nn.LayerNorm(channels)  # LayerNorm before feed-forward
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)  # GEGLU: value and gate
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)  # GEGLU output projection
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)  # 1x1 conv for output
    
    def forward(self, x, context):
        residue_long = x  # Save input for skip connection
        x = self.groupnorm(x)  # Normalize
        x = self.conv_input(x)  # 1x1 conv
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))  # Flatten spatial dims
        x = x.transpose(-1, -2)  # (n, h*w, c)
        residue_short = x  # Save for short skip
        x = self.layernorm_1(x)  # Normalize
        x = self.attention_1(x)  # Self-attention
        x += residue_short  # Add skip
        residue_short = x  # Update skip
        x = self.layernorm_2(x)  # Normalize
        x = self.attention_2(x, context)  # Cross-attention
        x += residue_short  # Add skip
        residue_short = x  # Update skip
        x = self.layernorm_3(x)  # Normalize
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)  # GEGLU: split value and gate
        x = x * F.gelu(gate)  # Gated activation
        x = self.linear_geglu_2(x)  # Project back
        x += residue_short  # Add skip
        x = x.transpose(-1, -2)  # (n, c, h*w)
        x = x.view((n, c, h, w))  # Restore spatial dims
        return self.conv_output(x) + residue_long  # Output with skip

class Upsample(nn.Module):
    """
    Upsampling block for UNet decoder.

    This block upsamples the spatial dimensions of the input feature map by a factor of 2 using nearest neighbor
    interpolation, followed by a convolution to refine the upsampled features.

    Args:
        channels (int): Number of input and output channels.

    Attributes:
        conv (nn.Conv2d): Convolutional layer applied after upsampling.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels, 2*H, 2*W).

    Notes:
        - Used in the decoder part of the UNet to progressively increase spatial resolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # Conv after upsampling
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # Nearest neighbor upsampling
        return self.conv(x)  # Refine with conv

class SwitchSequential(nn.Sequential):
    """
    Sequential container for UNet blocks with dynamic forward signatures.

    This container allows stacking of different types of blocks (e.g., residual, attention, upsample) that may require
    different arguments in their forward methods. It dispatches the correct arguments based on the block type.

    Args:
        *args: Sequence of nn.Module blocks to be applied sequentially.

    Forward Args:
        x (torch.Tensor): Input tensor.
        context (torch.Tensor): Context tensor for attention blocks.
        time (torch.Tensor): Time embedding tensor for residual blocks.

    Returns:
        torch.Tensor: Output tensor after sequentially applying all blocks.

    Notes:
        - Attention blocks receive (x, context).
        - Residual blocks receive (x, time).
        - Other blocks receive (x) only.
    """
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)  # Pass context to attention
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)     # Pass time to residual
            else:
                x = layer(x)           # Only x for other layers
        return x

class UNET(nn.Module):
    """
    UNet architecture for diffusion models.

    This class implements a hierarchical encoder-decoder architecture with skip connections, residual blocks,
    attention blocks, and upsampling. It is designed to process latent representations in diffusion models,
    enabling both local and global information flow.

    Args:
        None

    Attributes:
        encoders (nn.ModuleList): List of encoder blocks for downsampling and feature extraction.
        bottleneck (SwitchSequential): Bottleneck block at the lowest resolution.
        decoders (nn.ModuleList): List of decoder blocks for upsampling and reconstruction.

    Forward Args:
        x (torch.Tensor): Input latent tensor of shape (batch_size, 4, H, W).
        context (torch.Tensor): Context tensor for cross-attention (e.g., text embeddings).
        time (torch.Tensor): Time embedding tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 320, H, W).

    Notes:
        - Skip connections are used to concatenate encoder outputs with decoder inputs.
        - Attention blocks enable conditioning on external context.
        - Used as the core network in diffusion-based generative models.
    """
    def __init__(self):
        super().__init__()
        # Encoder: progressively downsample and extract features
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),  # Initial conv
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),  # Residual + attention
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
        # Bottleneck: process at lowest spatial resolution
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )
        # Decoder: upsample and reconstruct, using skip connections
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []  # Store outputs for skip connections
        for layers in self.encoders:
            x = layers(x, context, time)  # Pass through encoder block
            skip_connections.append(x)    # Save for skip connection
        x = self.bottleneck(x, context, time)  # Bottleneck
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)  # Concatenate skip connection
            x = layers(x, context, time)  # Pass through decoder block
        return x  # Final output

class UNET_OutputLayer(nn.Module):
    """
    Output layer for UNet in diffusion models.

    This layer normalizes the input feature map and projects it to the desired number of output channels using a
    convolution. It is typically used as the final layer to produce the model's output (e.g., predicted noise).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        groupnorm (nn.GroupNorm): Group normalization for input features.
        conv (nn.Conv2d): Convolutional layer for output projection.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W).

    Notes:
        - Used as the last layer in the diffusion model to produce the final prediction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)  # Normalize input
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Output conv
    
    def forward(self, x):
        x = self.groupnorm(x)  # Normalize
        x = F.silu(x)          # Activation
        x = self.conv(x)       # Project to output channels
        return x

class Diffusion(nn.Module):
    """
    Full diffusion model with time embedding, UNet, and output layer.

    This class combines the time embedding module, the main UNet architecture, and the output projection layer to
    form the complete diffusion model. It processes the input latent, conditions on context and time, and produces
    the final output.

    Args:
        None

    Attributes:
        time_embedding (TimeEmbedding): Module for embedding the diffusion time step.
        unet (UNET): Main UNet architecture for processing latents.
        final (UNET_OutputLayer): Output layer for projecting to desired channels.

    Forward Args:
        latent (torch.Tensor): Input latent tensor of shape (batch_size, 4, H, W).
        context (torch.Tensor): Context tensor for cross-attention.
        time (torch.Tensor): Time step tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 4, H, W).

    Notes:
        - This is the top-level model used for training and inference in diffusion-based generative models.
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)  # Embed time step
        self.unet = UNET()                        # Main UNet
        self.final = UNET_OutputLayer(320, 4)     # Output projection
    
    def forward(self, latent, context, time):
        time = self.time_embedding(time)          # Process time embedding
        output = self.unet(latent, context, time) # UNet forward
        output = self.final(output)               # Output projection
        return output
