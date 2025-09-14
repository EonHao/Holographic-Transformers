"""
Encoder components for the Holographic Transformer.

This file defines the encoder architecture and associated components for processing
complex-valued sequential data in the Holographic Transformer model.
"""

import torch
import torch.nn as nn
import math
import warnings
from typing import Optional

from .modules import ComplexLinear, ComplexLayerNorm, ComplexFFN, ComplexDropout, complex_kaiming_init
from .attention import HolographicMultiheadAttention


def get_complex_positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Create complex-valued positional encodings.

Generates sinusoidal-based positional encodings for complex numbers, where
positions are represented by unique complex exponentials that vary smoothly
across both position indices and feature dimensions.

Args:
    max_len: Maximum supported sequence length
    d_model: Feature dimensionality
    device: Device on which to allocate the tensor
    
Returns:
    Complex positional encoding tensor with shape [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model, dtype=torch.cfloat, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                        -(math.log(10000.0) / d_model))
    
    # Real part: cosine
    pe[:, 0::2].real = torch.cos(position * div_term)
    pe[:, 1::2].real = torch.cos(position * div_term)
    
    # Imaginary part: sine  
    pe[:, 0::2].imag = torch.sin(position * div_term)
    pe[:, 1::2].imag = torch.sin(position * div_term)
    
    return pe


class HoloTransformerEncoderLayer(nn.Module):
    """
    Single encoder layer for the Holographic Transformer.

Implements the pre-normalization architecture with holographic multi-head attention
followed by complex feed-forward networks, both with residual connections.

Args:
    d_model: Feature dimensionality
    n_heads: Number of parallel attention heads
    d_ff: Feed-forward network hidden layer size
    dropout: Dropout rate applied throughout the layer
    use_cosine_sim: Whether to use cosine similarity in attention calculations
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0,
                 use_cosine_sim: bool = True):
        super().__init__()
        self.d_model = d_model
        
        # Layer normalization
        self.ln1 = ComplexLayerNorm(d_model)
        self.ln2 = ComplexLayerNorm(d_model)
        
        # Holographic multi-head attention
        self.attention = HolographicMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_cosine_sim=use_cosine_sim
        )
        
        # Feed-forward network
        self.ffn = ComplexFFN(d_model, d_ff, dropout)
        
        self.dropout = ComplexDropout(dropout)
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through a single encoder layer.

Applies holographic attention followed by feed-forward processing with residual
connections and pre-normalization.

Args:
    x: Complex input tensor with shape [batch_size, seq_len, d_model]
    padding_mask: Boolean tensor indicating padding positions
    
Returns:
    Complex output tensor with the same shape as input
        """
        # Self-attention block with pre-normalization
        residual = x
        x = self.ln1(x)
        x = self.attention(x, padding_mask)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block with pre-normalization
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        
        return x
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}'


class HoloTransformerEncoder(nn.Module):
    """
    Encoder component for the Holographic Transformer.

Stacks multiple encoder layers and supports optional [CLS] token addition
and complex positional encodings to process sequential complex-valued data.

Args:
    d_model: Feature dimensionality
    n_heads: Number of attention heads per layer
    d_ff: Feed-forward network hidden layer size
    num_layers: Number of stacked encoder layers
    dropout: Dropout rate applied throughout the encoder
    use_cosine_sim: Whether to use cosine similarity in attention
    use_cls_token: Whether to prepend a [CLS] token for aggregation
    use_positional_encoding: Whether to apply positional information
    max_len: Maximum supported sequence length for positional encoding
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.0, use_cosine_sim: bool = True,
                 use_cls_token: bool = True, use_positional_encoding: bool = True,
                 max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.use_positional_encoding = use_positional_encoding
        self.max_len = max_len
        
        # [CLS] token embedding
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.cfloat))
            complex_kaiming_init(self.cls_token)
        
        # Positional encoding
        if use_positional_encoding:
            # We'll create this on the appropriate device in forward()
            self.register_buffer('_pos_encoding_created', torch.tensor(False))
        
        # Encoder layers
        self.layers = nn.ModuleList([
            HoloTransformerEncoderLayer(d_model, n_heads, d_ff, dropout, use_cosine_sim)
            for _ in range(num_layers)
        ])
        
        self.dropout = ComplexDropout(dropout)
    
    def _create_pos_encoding_if_needed(self, device: torch.device):
        """Generate and cache positional encoding when required."""
        if self.use_positional_encoding and not self._pos_encoding_created:
            pos_enc = get_complex_positional_encoding(self.max_len, self.d_model, device)
            self.register_buffer('pos_encoding', pos_enc)
            self._pos_encoding_created.fill_(True)
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input sequence through the full encoder stack.

Applies optional [CLS] token addition and positional encoding, then passes
through multiple stacked encoder layers.

Args:
    x: Complex input tensor with shape [batch_size, seq_len, d_model]
    padding_mask: Boolean tensor indicating padding positions
    
Returns:
    Complex encoded tensor with shape [batch_size, seq_len+1, d_model] if using CLS token,
    otherwise [batch_size, seq_len, d_model]
        """
        B, T, d_model = x.shape
        
        # Create positional encoding on correct device if needed
        self._create_pos_encoding_if_needed(x.device)
        
        # Add [CLS] token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]
            
            # Update padding mask
            if padding_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, T+1]
        
        # Add positional encoding
        if self.use_positional_encoding:
            seq_len = x.size(1)
            if seq_len <= self.max_len:
                pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
                x = x + pos_enc
            else:
                warnings.warn(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)
        
        return x
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_layers={len(self.layers)}, use_cls_token={self.use_cls_token}'
