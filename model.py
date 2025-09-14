"""
Holographic Transformer architecture implementation.

This file defines the complete HoloTransformer model that processes
complex-valued data while preserving and leveraging phase information through
holographic attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, NamedTuple

from .modules import ComplexLinear
from .encoder import HoloTransformerEncoder
from .decoder import DualHeadDecoder
from .losses import compute_total_loss


class HoloTransformerOutput(NamedTuple):
    """Output structure for HoloTransformer."""
    x_hat: torch.Tensor  # Reconstructed complex tensor [B, T, d_input]
    outputs: torch.Tensor  # Task outputs [B, num_outputs]
    loss_dict: Dict[str, torch.Tensor]  # Loss dictionary
    aux: Dict[str, Any]  # Auxiliary outputs (attention, phase differences)


class HoloTransformer(nn.Module):
    """
Holographic Transformer model designed for complex-valued data processing.
    
This main model class integrates encoders, holographic attention mechanisms,
and dual-output heads to handle complex inputs for both reconstruction and
analysis tasks while preserving phase relationships in the data.
    
Args:
    d_input: Input feature dimensionality
    d_model: Feature dimensionality throughout the model
    n_heads: Number of parallel attention heads
    d_ff: Hidden layer size in feedforward networks
    num_layers: Number of transformer layers in the encoder
    dropout: Dropout rate applied throughout the model
    use_cosine_sim: Whether to use cosine similarity in attention calculations
    use_cls_token: Whether to include a [CLS] token for task outputs
    lambda_recon: Weight coefficient for reconstruction loss
    lambda_task: Weight coefficient for task loss
    task_type: Type of task ('classification' or 'regression')
    num_classes: Number of classes for classification tasks
    num_outputs: Number of outputs for regression tasks
    use_positional_encoding: Whether to apply positional encoding
    max_len: Maximum sequence length supported
"""
    
    def __init__(self, d_input: int, d_model: int, n_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.0, use_cosine_sim: bool = True, use_cls_token: bool = True,
                 lambda_recon: float = 0.5, lambda_task: float = 1.0,
                 task_type: str = "classification", num_classes: int = 2, num_outputs: int = 1,
                 use_positional_encoding: bool = True, max_len: int = 5000):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.task_type = task_type
        self.lambda_recon = lambda_recon
        self.lambda_task = lambda_task
        
        # Input projection
        self.input_proj = ComplexLinear(d_input, d_model)
        
        # Encoder
        self.encoder = HoloTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            use_cosine_sim=use_cosine_sim,
            use_cls_token=use_cls_token,
            use_positional_encoding=use_positional_encoding,
            max_len=max_len
        )
        
        # Decoder
        self.decoder = DualHeadDecoder(
            d_model=d_model,
            d_input=d_input,
            task_type=task_type,
            num_classes=num_classes,
            num_outputs=num_outputs,
            use_cls_token=use_cls_token
        )
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None, return_attn: bool = False) -> HoloTransformerOutput:
        """
Process input through the HoloTransformer.
    
Args:
        x: Complex input tensor with shape [batch_size, seq_len, d_input]
        padding_mask: Boolean tensor indicating padding positions (True = padding)
        y: Target labels or values for loss computation (optional)
        return_attn: Whether to return attention weights and phase differences (optional)
        
Returns:
            HoloTransformerOutput object containing reconstructed input, task outputs,
            loss dictionary, and auxiliary information
"""
        # Input projection
        x_proj = self.input_proj(x)  # [B, T, d_model]
        
        # Encode
        encoded = self.encoder(x_proj, padding_mask)  # [B, T', d_model]
        
        # Decode
        x_hat, outputs = self.decoder(encoded, padding_mask)
        
        # Compute losses
        loss_dict = {}
        
        if y is not None:
            loss_dict = compute_total_loss(
                x_hat=x_hat,
                x=x,
                outputs=outputs,
                targets=y,
                task_type=self.task_type,
                lambda_recon=self.lambda_recon,
                lambda_task=self.lambda_task,
                padding_mask=padding_mask
            )
        else:
            # Compute only reconstruction loss if no targets provided
            from .losses import compute_reconstruction_loss
            recon_loss = compute_reconstruction_loss(x_hat, x, padding_mask)
            loss_dict['recon'] = recon_loss
        
        # Auxiliary outputs
        aux = {}
        if return_attn:
            # Get attention weights and phase differences from last encoder layer
            last_layer = self.encoder.layers[-1].attention
            aux['attn'] = last_layer.get_last_attention()
            aux['delta_phi'] = last_layer.get_last_delta_phi()
        
        return HoloTransformerOutput(
            x_hat=x_hat,
            outputs=outputs,
            loss_dict=loss_dict,
            aux=aux
        )
    
    def get_attention_weights(self, layer_idx: int = -1):
        """
        Get attention weights from a specific encoder layer.
        
        Args:
            layer_idx: Index of encoder layer (-1 for last layer)
            
        Returns:
            Attention weights from the specified layer
        """
        return self.encoder.layers[layer_idx].attention.get_last_attention()
    
    def get_phase_differences(self, layer_idx: int = -1):
        """
        Get phase differences from a specific encoder layer.
        
        Args:
            layer_idx: Index of encoder layer (-1 for last layer)
            
        Returns:
            Phase differences from the specified layer
        """
        return self.encoder.layers[layer_idx].attention.get_last_delta_phi()
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        return (f'd_input={self.d_input}, d_model={self.d_model}, '
                f'task_type={self.task_type}, num_params={self.count_parameters():,}')
