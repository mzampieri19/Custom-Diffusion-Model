import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """
    Embedding layer for token and position encoding in CLIP.

    This class provides token and position embeddings for input sequences, as well as layer normalization.
    It is used to convert input token indices into dense vector representations and add positional information.

    Args:
        n_vocab (int): Size of the vocabulary (number of unique tokens).
        n_embed (int): Dimensionality of the embedding vectors.
        n_tokens (int): Maximum number of tokens (sequence length).

    Attributes:
        token_embedding (nn.Embedding): Embedding layer mapping token indices to embedding vectors.
        position_embedding (nn.Parameter): Learnable positional embeddings for each position in the sequence.
        layernorm (nn.LayerNorm): Layer normalization applied to the embeddings.

    Forward Args:
        tokens (torch.LongTensor): Input tensor of token indices with shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Embedded input with positional encoding, shape (batch_size, sequence_length, n_embed).

    Notes:
        - The position embeddings are added to the token embeddings to provide positional information.
        - Layer normalization can help stabilize training and improve convergence.
    """
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        # Embedding layer for tokens
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        # Learnable positional embeddings for each position in the sequence
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # Convert token indices to embeddings
        x = self.token_embedding(tokens)
        # Add positional embeddings to token embeddings
        x += self.position_embedding
        # (Layer normalization is defined but not applied here)
        return x
    
class CLIPLayer(nn.Module):
    """
    Transformer encoder layer for CLIP.

    This class implements a single transformer encoder block, consisting of layer normalization, multi-head self-attention,
    and a feed-forward network with a gated activation. Residual connections are used around both the attention and feed-forward sublayers.

    Args:
        n_heads (int): Number of attention heads in the self-attention mechanism.
        n_embed (int): Dimensionality of the input and output embeddings.

    Attributes:
        layernorm_1 (nn.LayerNorm): Layer normalization before self-attention.
        attention (SelfAttention): Multi-head self-attention module.
        layernorm_2 (nn.LayerNorm): Layer normalization before the feed-forward network.
        linear_1 (nn.Linear): First linear layer in the feed-forward network, expands dimensionality.
        linear_2 (nn.Linear): Second linear layer in the feed-forward network, projects back to embedding size.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embed).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embed).

    Notes:
        - Uses residual connections around both the attention and feed-forward sublayers.
        - The feed-forward network uses a gated activation (GLU-like).
        - This layer is typically stacked multiple times in the model.
    """
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        # Layer normalization before self-attention
        self.layernorm_1 = nn.LayerNorm(n_embed)
        # Multi-head self-attention module
        self.attention = SelfAttention(n_heads, n_embed)
        # Layer normalization before feed-forward network
        self.layernorm_2 = nn.LayerNorm(n_embed)
        # First linear layer in feed-forward network (expands dimensionality)
        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        # Second linear layer in feed-forward network (projects back to embedding size)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection
        residue = x
        # Apply layer normalization before attention
        x = self.layernorm_1(x)
        # Apply multi-head self-attention with causal mask
        x = self.attention(x, causal_mask=True)
        # Add residual connection after attention
        x += residue
        # Save for next residual connection
        residue = x
        # Apply layer normalization before feed-forward
        x = self.layernorm_2(x)
        # Feed-forward network: first linear layer
        x = self.linear_1(x)
        # Gated Linear Unit (GLU)-like activation
        x = x * torch.sigmoid(1.702 * x)
        # Second linear layer in feed-forward network
        x = self.linear_2(x)
        # Add residual connection after feed-forward
        x += residue
        return x 

class CLIP(nn.Module):
    """
    CLIP transformer encoder model.

    This class implements the main CLIP transformer encoder, consisting of an embedding layer, a stack of transformer layers,
    and a final layer normalization. It processes input token sequences and outputs contextualized embeddings.

    Args:
        None (parameters are hardcoded for the CLIP model).

    Attributes:
        embedding (CLIPEmbedding): Embedding layer for tokens and positions.
        layers (nn.ModuleList): List of stacked CLIPLayer transformer blocks.
        layersnorm (nn.LayerNorm): Final layer normalization.

    Forward Args:
        x (torch.LongTensor): Input tensor of token indices with shape (batch_size, sequence_length).

    Returns:
        torch.FloatTensor: Output tensor of contextualized embeddings, shape (batch_size, sequence_length, 768).

    Notes:
        - The model uses 12 transformer layers and an embedding size of 768.
        - Designed for use in the CLIP architecture for vision-language tasks.
    """
    def __init__(self):
        super().__init__()
        # Embedding layer for tokens and positions
        self.embedding = CLIPEmbedding(49408, 768, 77)
        # Stack of 12 transformer encoder layers
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range (12)
        ])
        # Final layer normalization
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Ensure input is of type long (token indices)
        tokens = tokens.type(torch.long)
        # Get token and position embeddings
        state = self.embedding(tokens)
        # Pass through each transformer encoder layer
        for layer in self.layers:
            state = layer(state)
        # Apply final layer normalization
        output = self.layernorm(state)
        return output