"""
Holographic Multi-Head Attention module.

This file defines the key attention mechanism that powers the Holographic Transformer. 
It extends standard self-attention with complex-valued operations, including phase differences,
coherence decay, and coherent superposition to better model phase information in complex signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .modules import ComplexLinear
from .utils import wrap_angle


class HolographicMultiheadAttention(nn.Module):
    """
    Multi-head attention mechanism with holographic phase interference.
    
    This implementation introduces complex inner products, phase-based attention modulation,
    and coherent superposition to capture interference patterns in complex-valued data.
    
    The key steps include:
    1. Computing complex inner products between queries and keys
    2. Calculating phase differences between query-key pairs
    3. Applying coherence decay based on phase differences
    4. Performing coherent superposition of values with phase rotation
    
    Args:
        d_model: Dimensionality of the input and output features
        n_heads: Number of parallel attention heads
        dropout: Dropout rate applied to attention weights
        use_cosine_sim: Whether to use cosine similarity (normalizes for magnitude)
        eps: Small value added for numerical stability in divisions
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_cosine_sim: bool = True, eps: float = 1e-8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_cosine_sim = use_cosine_sim
        self.eps = eps
        
        # Linear projections for Q, K, V
        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)
        
        # Learnable coherence decay parameter α
        # Use softplus to ensure α > 0
        self.alpha_param = nn.Parameter(torch.zeros(n_heads))
        
        # Note: We use regular dropout for attention weights (which are real)
        self.dropout = nn.Dropout(dropout)
        
        # Store last attention weights and phase differences for visualization
        self.last_attention = None
        self.last_delta_phi = None
        
    
    def get_last_attention(self) -> Optional[torch.Tensor]:
        """Return the attention weights from the most recent forward pass."""
        return self.last_attention
    
    def get_last_delta_phi(self) -> Optional[torch.Tensor]:
        """Return the phase differences from the most recent forward pass."""
        return self.last_delta_phi
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process complex-valued input through holographic attention.
    
    Args:
        x: Complex input tensor with shape [batch_size, seq_len, d_model]
        padding_mask: Boolean tensor indicating padding positions (True = padding)
        
    Returns:
        Complex output tensor with the same shape as input
        """
        B, T, d_model = x.shape
        
        # Linear projections
        Q = self.q_proj(x)  # [B, T, d_model]
        K = self.k_proj(x)  # [B, T, d_model]
        V = self.v_proj(x)  # [B, T, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        
        # Compute complex inner products s_ij = <Q_i, K_j>_c
        # s_ij = sum_d Q_i[d] * conj(K_j[d])
        s = torch.einsum('bhid,bhjd->bhij', Q, K.conj())  # [B, H, T, T]
        
        # Compute phase differences Δφ_ij = angle(s_ij) 
        # Using atan2 for numerical stability
        delta_phi = torch.atan2(s.imag + self.eps, s.real + self.eps)  # [B, H, T, T]
        delta_phi = wrap_angle(delta_phi)  # Wrap to (-π, π]
        
        # Store for visualization
        self.last_delta_phi = delta_phi.detach()
        
        # Compute similarity scores
        if self.use_cosine_sim:
            # Cosine version: sim_ij = Re(s_ij) / (||Q_i|| * ||K_j|| + eps)
            q_norm = torch.norm(Q, dim=-1, keepdim=True)  # [B, H, T, 1]
            k_norm = torch.norm(K, dim=-1, keepdim=True)  # [B, H, T, 1]
            # Broadcast norms to compute pairwise products
            norms = q_norm * k_norm.transpose(-2, -1)  # [B, H, T, T]
            sim = s.real / (norms + self.eps)
        else:
            # Dot version: sim_ij = Re(s_ij)
            sim = s.real
        
        # Apply coherence decay: W_ij = sim_ij / sqrt(d_k) * exp(-α * |Δφ_ij|)
        alpha = F.softplus(self.alpha_param) + 1e-6  # Ensure α > 0
        alpha = alpha.view(1, self.n_heads, 1, 1)  # [1, H, 1, 1]
        
        coherence_decay = torch.exp(-alpha * torch.abs(delta_phi))
        
        weights = (sim / math.sqrt(self.d_k)) * coherence_decay  # [B, H, T, T]
        
        # Apply padding mask
        if padding_mask is not None:
            # padding_mask: [B, T], True indicates padding
            mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            weights = weights.masked_fill(mask, -1e9)
        
        # Softmax to get attention probabilities
        attn = F.softmax(weights, dim=-1)  # [B, H, T, T]
        attn = self.dropout(attn)
        
        # Store for visualization
        self.last_attention = attn.detach()
        
        # Coherent superposition: H_i = sum_j α_ij * (V_j * exp(i * Δφ_ij))
        # Apply phase rotation to values
        phasor = torch.exp(1j * delta_phi)  # [B, H, T, T]
        # For each query position i, rotate all value positions j by phase difference
        out = torch.zeros_like(V)  # [B, H, T, d_k]
        for i in range(T):
            # Get attention weights and phasors for query i
            attn_i = attn[:, :, i, :]  # [B, H, T]
            phasor_i = phasor[:, :, i, :]  # [B, H, T]
            
            # Apply phase rotation to all values and weight by attention
            rotated_values = V * phasor_i.unsqueeze(-1)  # [B, H, T, d_k]
            out[:, :, i, :] = torch.sum(attn_i.unsqueeze(-1) * rotated_values, dim=2)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # [B, T, d_model]
        out = self.out_proj(out)
        
        return out
    
    def extra_repr(self) -> str:
        """Return string representation with key parameters."""
        return f'd_model={self.d_model}, n_heads={self.n_heads}, use_cosine_sim={self.use_cosine_sim}'
