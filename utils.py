"""
Utility functions and helpers for Holographic Transformer implementation.

This file provides utility functions for angle normalization, random seed
management, gradient computation, synthetic data generation, and device handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import math
import warnings
from typing import Tuple


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize angles to the standard range of (-π, π].

Uses arctangent with sine and cosine to ensure angles fall within the
standard mathematical range for trigonometric calculations.

Args:
    angle: Tensor containing angle values in radians
    
Returns:
    Tensor with angles wrapped to the range (-π, π]
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def set_seed(seed: int = 42) -> None:
    """Configure random number generators for consistent results across runs.

Sets seeds for PyTorch, NumPy, and CUDA (if available) to ensure
experiment reproducibility.

Args:
    seed: Integer value to use as the base random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Use deterministic algorithms when possible
        try:
            torch.use_deterministic_algorithms(True)
        except:
            warnings.warn("Deterministic algorithms not available on this platform")


def get_gradient_norm(model: nn.Module) -> float:
    """Compute the L2 norm of gradients across all model parameters.

Calculates the root mean square of parameter gradients to monitor
training stability and help with gradient clipping decisions.

Args:
    model: PyTorch model whose gradients to evaluate
    
Returns:
    Floating-point value representing the total gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)


class SyntheticComplexDataset(Dataset):
    """
    Synthetic dataset generator for complex-valued data.

Creates a dataset with two distinct classes based on phase relationships:
- Class 0: Adjacent elements have similar phases (promoting constructive interference)
- Class 1: Adjacent elements have opposite phases (creating destructive interference)

Args:
    num_samples: Total number of samples to generate
    seq_len: Length of each sequence in tokens
    d_input: Dimensionality of the complex embedding space
    noise_level: Magnitude of random noise to add to the data
    device: PyTorch device to store the generated tensors on
    """
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 32, d_input: int = 16,
                 noise_level: float = 0.1, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.d_input = d_input
        self.device = device
        
        # Generate data
        self.data, self.labels = self._generate_data(noise_level)
    
    def _generate_data(self, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create the dataset samples with controlled phase relationships.

Generates complex-valued sequences where the phase relationships between
tokens determine the class label.

Args:
    noise_level: Amount of Gaussian noise to add to magnitudes and phases
    
Returns:
    Tuple containing (complex_data_tensor, class_labels_tensor)"""
        data = torch.zeros(self.num_samples, self.seq_len, self.d_input, dtype=torch.cfloat)
        labels = torch.zeros(self.num_samples, dtype=torch.long)
        
        for i in range(self.num_samples):
            # Random class
            class_label = torch.randint(0, 2, (1,)).item()
            labels[i] = class_label
            
            # Base magnitude and phase
            base_magnitude = torch.rand(self.d_input) * 2 + 0.5  # [0.5, 2.5]
            base_phase = torch.rand(self.d_input) * 2 * math.pi - math.pi  # [-π, π]
            
            for t in range(self.seq_len):
                if class_label == 0:
                    # Class 0: Similar phases (small phase differences)
                    phase_offset = torch.randn(self.d_input) * 0.2  # Small phase variation
                else:
                    # Class 1: Alternating phases (large phase differences)
                    if t % 2 == 0:
                        phase_offset = torch.zeros(self.d_input)
                    else:
                        phase_offset = torch.ones(self.d_input) * math.pi  # π phase shift
                
                # Create complex values
                magnitude = base_magnitude * (1 + torch.randn(self.d_input) * noise_level)
                phase = base_phase + phase_offset + torch.randn(self.d_input) * noise_level
                
                data[i, t] = magnitude * torch.exp(1j * phase)
        
        return data.to(self.device), labels.to(self.device)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_device() -> torch.device:
    """Determine the optimal computing device for model operations.

Prioritizes CUDA GPUs if available, then MPS (Apple Silicon),
falling back to CPU if neither accelerator is available.

Returns:
    PyTorch device object representing the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Calculate the total number of trainable parameters in a model.

Useful for model size comparison and memory planning.

Args:
    model: PyTorch model to analyze
    
Returns:
    Integer count of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def complex_to_real_params(num_complex_params: int) -> int:
    """Calculate the number of real parameters equivalent to complex parameters.

Since each complex parameter consists of separate real and imaginary
components, this function doubles the count.

Args:
    num_complex_params: Number of complex-valued parameters
    
Returns:
    Equivalent number of real-valued parameters"""
    return num_complex_params * 2  # Each complex param = 2 real params
