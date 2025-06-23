import torch
from torch import nn
from torch.nn import functional as F
import math

# SelfAttention implements multi-head self-attention for a sequence of embeddings.
class SelfAttention(nn.Module):
    """
    Multi-head self-attention module for sequence modeling.

    This class implements the self-attention mechanism, a core component of Transformer-based architectures.
    Self-attention allows each position in the input sequence to attend to all other positions, enabling the model
    to capture dependencies regardless of their distance in the sequence.

    Args:
        n_heads (int): Number of attention heads. Each head learns different attention patterns.
        d_embed (int): Dimensionality of the input embeddings. Must be divisible by n_heads.
        in_proj_bias (bool, optional): If True, adds a bias term to the input projection layer. Default is True.
        out_proj_bias (bool, optional): If True, adds a bias term to the output projection layer. Default is True.

    Attributes:
        in_proj (nn.Linear): Linear layer that projects the input embeddings into concatenated queries, keys, and values.
            This layer outputs a tensor of shape (batch_size, sequence_length, 3 * d_embed), which is then split into
            separate query, key, and value tensors for each attention head.
        out_proj (nn.Linear): Linear layer that projects the concatenated outputs of all attention heads back to the
            original embedding dimension. This allows the multi-head attention outputs to be combined and integrated
            into the model.
        n_heads (int): Number of attention heads used in the multi-head attention mechanism.
        d_head (int): Dimensionality of each attention head, computed as d_embed // n_heads.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_embed), representing a batch of sequences.
        causal_mask (bool, optional): If True, applies a causal mask to prevent each position from attending to future
            positions. This is typically used in autoregressive models (e.g., language modeling) to preserve causality.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_embed), containing the attended representations
        for each position in the input sequence.

    Notes:
        - The self-attention mechanism computes attention weights between all pairs of positions in the sequence,
          optionally applying a causal mask for autoregressive tasks.
        - Multi-head attention enables the model to jointly attend to information from different representation subspaces.
        - This module is typically used as a building block in larger models such as Transformers, where it is combined
          with feed-forward layers, normalization, and residual connections.
    """
    
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Linear layer to project input to queries, keys, and values
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Linear layer for output projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads  # Dimension per attention head

    def forward(self, x, causal_mask=False):
        # x: (batch_size, sequence_length, d_embed)
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # Project input to queries, keys, and values and split into heads
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention weights
        weight = q @ k.transpose(-1, -2)  # (batch, n_heads, seq_len, seq_len)
        if causal_mask:
            # Apply causal mask to prevent attending to future positions
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        weight /= math.sqrt(self.d_head)  # Scale by sqrt(d_head)
        weight = F.softmax(weight, dim=-1)  # Softmax over keys

        # Weighted sum of values
        output = weight @ v  # (batch, n_heads, seq_len, d_head)
        output = output.transpose(1, 2)  # (batch, seq_len, n_heads, d_head)
        output = output.reshape(input_shape)  # (batch, seq_len, d_embed)
        output = self.out_proj(output)  # Final linear projection
        return output

class CrossAttention(nn.Module):
    """
        Multi-head cross-attention module for sequence modeling.
        This class implements the cross-attention mechanism, which allows each position in a query sequence to attend to all positions in a separate context (key/value) sequence. 
        Cross-attention is a core component in architectures such as encoder-decoder Transformers and diffusion models, enabling information flow between different modalities or representations.
        Args:
            n_heads (int): Number of attention heads. Each head learns different attention patterns.
            d_embed (int): Dimensionality of the query embeddings. Must be divisible by n_heads.
            d_cross (int): Dimensionality of the context (key/value) embeddings.
            in_proj_bias (bool, optional): If True, adds a bias term to the input projection layers. Default is True.
            out_proj_bias (bool, optional): If True, adds a bias term to the output projection layer. Default is True.
        Attributes:
            q_proj (nn.Linear): Linear layer projecting query inputs to queries.
            k_proj (nn.Linear): Linear layer projecting context inputs to keys.
            v_proj (nn.Linear): Linear layer projecting context inputs to values.
            out_proj (nn.Linear): Linear layer projecting the concatenated outputs of all attention heads back to the original embedding dimension.
            n_heads (int): Number of attention heads.
            d_head (int): Dimensionality of each attention head, computed as d_embed // n_heads.
        Forward Args:
            x (torch.Tensor): Query tensor of shape (batch_size, sequence_length, d_embed).
            y (torch.Tensor): Context tensor (for keys and values) of shape (batch_size, sequence_length, d_cross).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_embed), containing the attended representations for each position in the query sequence.
        Notes:
            - The cross-attention mechanism computes attention weights between all pairs of positions in the query and context sequences.
            - Multi-head attention enables the model to jointly attend to information from different representation subspaces.
            - This module is typically used in encoder-decoder architectures, cross-modal models, and diffusion models for conditioning on external information.
    """
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Linear projections for queries, keys, and values
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Output projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads  # Dimension per attention head
    
    def forward(self, x, y):
        # x: (batch_size, sequence_length, d_embed) - queries
        # y: (batch_size, sequence_length, d_cross) - keys/values
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        q = q.view(interim_shape).transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention weights
        weight = q @ k.transpose(-1, -2)  # (batch, n_heads, seq_len, seq_len)
        weight /= math.sqrt(self.d_head)  # Scale by sqrt(d_head)
        weight = F.softmax(weight, dim=-1)  # Softmax over keys

        # Weighted sum of values
        output = weight @ v  # (batch, n_heads, seq_len, d_head)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_head)
        output = output.view(input_shape)  # (batch, seq_len, d_embed)
        output = self.out_proj(output)  # Final linear projection
        return output
