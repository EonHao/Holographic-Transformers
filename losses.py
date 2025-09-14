"""
Loss functions for training Holographic Transformers.

This file defines specialized loss functions for training the Holographic Transformer,
including complex-valued reconstruction loss and task-specific loss components.
"""

import torch
import torch.nn as nn
from typing import Dict


def compute_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, 
                              padding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate reconstruction loss for complex-valued tensors.

Measures the reconstruction error by combining the squared differences
of both real and imaginary components between predicted and target tensors.

Args:
    x_hat: Reconstructed complex tensor with shape [batch_size, seq_len, d_input]
    x: Original complex tensor with shape [batch_size, seq_len, d_input]
    padding_mask: Optional boolean tensor indicating padding positions
    
Returns:
    Scalar tensor representing the reconstruction loss value
    """
    # Compute complex difference
    diff = x_hat - x  # [B, T, d_input]
    
    # Sum of squared errors for real and imaginary parts
    real_loss = torch.mean(diff.real ** 2)
    imag_loss = torch.mean(diff.imag ** 2)
    
    # If padding mask is provided, exclude padded positions
    if padding_mask is not None:
        # Create mask for valid positions
        valid_mask = ~padding_mask  # [B, T]
        valid_mask = valid_mask.unsqueeze(-1)  # [B, T, 1]
        
        # Masked loss computation
        real_loss = torch.sum((diff.real ** 2) * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        imag_loss = torch.sum((diff.imag ** 2) * valid_mask) / (torch.sum(valid_mask) + 1e-8)
    
    return real_loss + imag_loss


def compute_task_loss(outputs: torch.Tensor, targets: torch.Tensor, 
                     task_type: str) -> torch.Tensor:
    """
    Calculate loss for the analysis task.

Applies the appropriate loss function based on whether the task is
classification or regression.

Args:
    outputs: Model predictions for the analysis task
    targets: Ground truth labels or values
    task_type: Type of task ('classification' or 'regression')
    
Returns:
    Scalar tensor representing the task-specific loss value
    """
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, targets)
    elif task_type == "regression":
        loss_fn = nn.MSELoss()
        return loss_fn(outputs, targets)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def compute_total_loss(x_hat: torch.Tensor, x: torch.Tensor, 
                      outputs: torch.Tensor, targets: torch.Tensor,
                      task_type: str, lambda_recon: float = 0.5, 
                      lambda_task: float = 1.0,
                      padding_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss function for end-to-end training.

Calculates the total training loss by combining reconstruction loss and task loss
with specified weighting factors.

Args:
    x_hat: Reconstructed complex tensor
    x: Original complex tensor
    outputs: Model predictions for the analysis task
    targets: Ground truth labels or values
    task_type: Type of task ('classification' or 'regression')
    lambda_recon: Weight coefficient for the reconstruction loss component
    lambda_task: Weight coefficient for the task loss component
    padding_mask: Optional boolean tensor indicating padding positions
    
Returns:
    Dictionary containing individual and combined loss values
    """
    recon_loss = compute_reconstruction_loss(x_hat, x, padding_mask)
    task_loss = compute_task_loss(outputs, targets, task_type)
    total_loss = lambda_recon * recon_loss + lambda_task * task_loss
    
    return {
        'recon': recon_loss,
        'task': task_loss, 
        'total': total_loss
    }
