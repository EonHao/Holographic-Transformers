"""
Holographic Transformers for Complex-Valued Signal Processing
============================================================

  _   _       _     _____                     __                                
 | | | | ___ | | __|_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __ 
 | |_| |/ _ \| |/ _ \| || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
 |  _  | (_) | | (_) | || | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |   
 |_| |_|\___/|_|\___/|_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_|   
                                                                                

 ============================================================

A PyTorch implementation of Holographic Transformers that integrate 
phase interference into self-attention mechanisms for complex-valued 
signal processing.

Main Components:
- ComplexLinear: Complex linear layers with proper initialization
- ComplexLayerNorm: Layer normalization for complex tensors  
- HolographicMultiheadAttention: Core attention with phase interference
- HoloTransformerEncoder: Multi-layer encoder with complex positional encoding
- DualHeadDecoder: Reconstruction and analysis heads
- HoloTransformer: Complete model

Author: AI Assistant
"""

from .modules import (
    ComplexLinear,
    ComplexLayerNorm, 
    ComplexFFN,
    ComplexDropout,
    complex_gelu,
    modrelu,
    complex_kaiming_init
)

from .attention import HolographicMultiheadAttention

from .encoder import (
    HoloTransformerEncoderLayer,
    HoloTransformerEncoder,
    get_complex_positional_encoding
)

from .decoder import DualHeadDecoder

from .model import HoloTransformer, HoloTransformerOutput

from .losses import compute_reconstruction_loss, compute_task_loss

from .utils import (
    wrap_angle,
    set_seed,
    get_gradient_norm,
    SyntheticComplexDataset
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    # Core modules
    "ComplexLinear",
    "ComplexLayerNorm", 
    "ComplexFFN",
    "ComplexDropout",
    "complex_gelu",
    "modrelu",
    "complex_kaiming_init",
    
    # Attention
    "HolographicMultiheadAttention",
    
    # Encoder
    "HoloTransformerEncoderLayer",
    "HoloTransformerEncoder", 
    "get_complex_positional_encoding",
    
    # Decoder
    "DualHeadDecoder",
    
    # Main model
    "HoloTransformer",
    "HoloTransformerOutput",
    
    # Losses
    "compute_reconstruction_loss",
    "compute_task_loss",
    
    # Utils
    "wrap_angle",
    "set_seed", 
    "get_gradient_norm",
    "SyntheticComplexDataset",
]
