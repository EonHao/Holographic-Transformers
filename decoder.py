"""
Dual-head decoder component for the Holographic Transformer.

This file defines a decoder with two parallel heads: one for reconstructing the
original input and another for performing analysis tasks like classification or regression
on complex-encoded representations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .modules import ComplexLinear


class DualHeadDecoder(nn.Module):
    """
    Dual-output decoder for joint reconstruction and analysis tasks.

This decoder processes encoded complex representations to perform two parallel operations:
1. Reconstruct the original input from encoded representations
2. Perform classification or regression on aggregated representation features

For analysis tasks, complex features are converted to real values by concatenating
their real and imaginary components before processing through a feed-forward network.

Args:
    d_model: Feature dimensionality of the encoded representations
    d_input: Original input feature dimensionality for reconstruction
    task_type: Type of analysis task ('classification' or 'regression')
    num_classes: Number of target classes for classification tasks
    num_outputs: Number of output values for regression tasks
    use_cls_token: Whether the encoder includes a [CLS] token for aggregation
    """
    
    def __init__(self, d_model: int, d_input: int, task_type: str = "classification",
                 num_classes: int = 2, num_outputs: int = 1, use_cls_token: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.task_type = task_type
        self.use_cls_token = use_cls_token
        
        # Reconstruction head
        self.recon_head = ComplexLinear(d_model, d_input)
        
        # Analysis head
        if task_type == "classification":
            self.num_outputs = num_classes
        elif task_type == "regression":
            self.num_outputs = num_outputs
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Complex-to-real mapping: [Re, Im] -> ReLU -> outputs
        self.analysis_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for real and imaginary parts
            nn.ReLU(),
            nn.Linear(d_model, self.num_outputs)
        )
    
    def forward(self, encoded: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process encoded representations through both decoder heads.

Reconstructs the input data and performs analysis tasks in parallel using
the encoded complex representations from the encoder.

Args:
    encoded: Complex encoded tensor with shape [batch_size, seq_len, d_model]
             or [batch_size, seq_len+1, d_model] if using CLS token
    padding_mask: Boolean tensor indicating padding positions
    
Returns:
    Tuple containing:
    - x_hat: Reconstructed complex input with shape [batch_size, seq_len, d_input]
    - outputs: Analysis task outputs with shape [batch_size, num_outputs]
        """
        B, seq_len, d_model = encoded.shape
        
        if self.use_cls_token:
            # Split [CLS] token and sequence
            cls_token = encoded[:, 0]  # [B, d_model]
            sequence = encoded[:, 1:]  # [B, T, d_model]
            
            # Use [CLS] token for analysis
            pooled = cls_token
        else:
            sequence = encoded
            
            # Mask-aware mean pooling for analysis
            if padding_mask is not None:
                # Compute valid lengths
                valid_mask = ~padding_mask  # [B, T]
                valid_lengths = valid_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                
                # Masked mean
                masked_sequence = sequence * valid_mask.unsqueeze(-1)  # [B, T, d_model]
                pooled = masked_sequence.sum(dim=1) / (valid_lengths.unsqueeze(-1) + 1e-8)  # [B, d_model]
            else:
                # Simple mean pooling
                pooled = sequence.mean(dim=1)  # [B, d_model]
        
        # Reconstruction head
        x_hat = self.recon_head(sequence)  # [B, T, d_input]
        
        # Analysis head: complex to real
        pooled_real = torch.cat([pooled.real, pooled.imag], dim=-1)  # [B, d_model*2]
        outputs = self.analysis_head(pooled_real)  # [B, num_outputs]
        
        return x_hat, outputs
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, d_input={self.d_input}, task_type={self.task_type}, num_outputs={self.num_outputs}'
