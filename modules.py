"""
Core complex-valued neural network modules.

This module contains the fundamental building blocks for complex-valued
neural networks including linear layers, normalization, and activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple


def complex_kaiming_init(tensor: torch.Tensor, gain: float = 1.0) -> None:
    """
    Kaiming-like initialization for complex tensors.
    Initialize magnitude with Kaiming normal and phase uniformly.
    
    Args:
        tensor: Complex tensor to initialize
        gain: Scaling factor
    """
    with torch.no_grad():
        # Get fan_in for proper scaling
        if tensor.dim() >= 2:
            fan_in = tensor.size(1) * tensor.numel() // (tensor.size(0) * tensor.size(1))
        else:
            fan_in = tensor.size(0)
        
        # Kaiming normal for magnitude
        std = gain * math.sqrt(2.0 / fan_in)
        magnitude = torch.randn_like(tensor.real) * std
        
        # Uniform phase
        phase = torch.rand_like(tensor.real) * 2 * math.pi - math.pi
        
        # Set complex values
        tensor.real.copy_(magnitude * torch.cos(phase))
        tensor.imag.copy_(magnitude * torch.sin(phase))


class ComplexLinear(nn.Module):
    """
    Complex linear layer implemented as real/imaginary separate paths.
    
    Performs complex matrix multiplication: y = Wx + b where W and b are complex.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension  
        bias: Whether to include bias term
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Complex weight W = W_real + i * W_imag
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.cfloat))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with complex Kaiming initialization."""
        complex_kaiming_init(self.weight)
        if self.bias is not None:
            with torch.no_grad():
                bound = 1 / math.sqrt(self.in_features)
                magnitude = torch.rand_like(self.bias.real) * bound
                phase = torch.rand_like(self.bias.real) * 2 * math.pi - math.pi
                self.bias.real.copy_(magnitude * torch.cos(phase))
                self.bias.imag.copy_(magnitude * torch.sin(phase))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
    Apply complex linear transformation to input tensor.

    Performs complex matrix multiplication using the layer's weight and bias parameters.

    Args:
        input: Complex input tensor with shape [..., in_features]
        
    Returns:
        Complex output tensor with shape [..., out_features]
    """
        # Complex matrix multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        output = F.linear(input, self.weight, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class ComplexLayerNorm(nn.Module):
    """
    Complex layer normalization - normalize real and imaginary parts separately.
    
    This applies layer normalization to the real and imaginary parts independently,
    with optional shared or separate affine parameters.
    
    Args:
        normalized_shape: Shape to normalize over
        affine: Whether to include learnable affine parameters
        shared_affine: Whether to share affine parameters between real/imaginary parts
        eps: Small constant for numerical stability
    """
    
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], 
                 affine: bool = True, shared_affine: bool = False, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.affine = affine
        self.shared_affine = shared_affine
        self.eps = eps
        
        if affine:
            if shared_affine:
                # Shared parameters for real and imaginary parts
                self.weight = nn.Parameter(torch.ones(normalized_shape))
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                # Separate parameters for real and imaginary parts
                self.weight_real = nn.Parameter(torch.ones(normalized_shape))
                self.bias_real = nn.Parameter(torch.zeros(normalized_shape))
                self.weight_imag = nn.Parameter(torch.ones(normalized_shape))
                self.bias_imag = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            if not shared_affine:
                self.register_parameter('weight_real', None)
                self.register_parameter('bias_real', None)
                self.register_parameter('weight_imag', None)
                self.register_parameter('bias_imag', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
    Apply layer normalization to complex input data.

    Processes real and imaginary components separately using standard layer normalization,
    maintaining the complex structure of the output.

    Args:
        input: Complex input tensor to normalize
        
    Returns:
        Complex tensor with normalized real and imaginary parts
    """
        # Normalize real and imaginary parts separately
        real_part = input.real
        imag_part = input.imag
        
        # Layer norm on real part
        real_normalized = F.layer_norm(real_part, self.normalized_shape, eps=self.eps)
        # Layer norm on imaginary part  
        imag_normalized = F.layer_norm(imag_part, self.normalized_shape, eps=self.eps)
        
        if self.affine:
            if self.shared_affine:
                real_normalized = real_normalized * self.weight + self.bias
                imag_normalized = imag_normalized * self.weight + self.bias
            else:
                real_normalized = real_normalized * self.weight_real + self.bias_real
                imag_normalized = imag_normalized * self.weight_imag + self.bias_imag
        
        return torch.complex(real_normalized, imag_normalized)
    
    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, affine={self.affine}, shared_affine={self.shared_affine}, eps={self.eps}'


def complex_gelu(z: torch.Tensor) -> torch.Tensor:
    """
    Complex GELU activation - apply GELU to real and imaginary parts separately.
    
    Args:
        z: Complex input tensor
        
    Returns:
        Complex output with GELU applied to real/imaginary parts
    """
    return torch.complex(F.gelu(z.real), F.gelu(z.imag))


class ComplexDropout(nn.Module):
    """
    Dropout for complex tensors.
    
    Applies the same dropout mask to both real and imaginary parts.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return input
        
        # Generate dropout mask based on the magnitude
        mask = torch.rand_like(input.real) > self.p
        # Apply same mask to both real and imaginary parts
        scale = 1.0 / (1 - self.p)  # Scale to maintain expected value
        return input * mask.to(input.dtype) * scale


def modrelu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    ModReLU activation: ReLU(|z| + bias) * z/|z|
    
    This activation preserves the phase while applying ReLU to the magnitude.
    
    Args:
        z: Complex input tensor
        bias: Bias parameter
        
    Returns:
        ModReLU activated complex tensor
    """
    magnitude = torch.abs(z)
    activated_mag = F.relu(magnitude + bias)
    
    # Avoid division by zero
    safe_magnitude = magnitude + 1e-8
    return activated_mag * z / safe_magnitude


class ComplexFFN(nn.Module):
    """
    Complex Feed-Forward Network.
    
    A two-layer feed-forward network with complex linear layers and 
    complex activation functions.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
        activation: Activation function ("gelu" or "modrelu")
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, 
                 activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        self.linear1 = ComplexLinear(d_model, d_ff)
        self.linear2 = ComplexLinear(d_ff, d_model)
        self.dropout = ComplexDropout(dropout)
        
        if activation == "modrelu":
            self.modrelu_bias = nn.Parameter(torch.zeros(d_ff))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
    Process input through complex feed-forward network.

    Applies sequential complex linear transformations with activation and dropout
    between layers to introduce non-linearity while preserving complex structure.

    Args:
        x: Complex input tensor with shape [batch_size, seq_len, d_model]
        
    Returns:
        Complex output tensor with the same shape as input
    """
        # First linear layer
        x = self.linear1(x)
        
        # Activation
        if self.activation == "gelu":
            x = complex_gelu(x)
        elif self.activation == "modrelu":
            x = modrelu(x, self.modrelu_bias)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Dropout
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, d_ff={self.d_ff}, activation={self.activation}'
