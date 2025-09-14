
"""
Attention and phase visualization tools for Holographic Transformers.

This script provides utilities for analyzing and visualizing attention patterns,
phase differences, and interference effects in the Holographic Transformer model.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import HoloTransformer, set_seed, SyntheticComplexDataset
from holo_transformer.utils import get_device

# Set device and seed
device = get_device()
set_seed(42)


def create_visualization_data():
    """Generate structured complex-valued data with controlled phase patterns.

Creates input data with specific phase relationships to demonstrate
how the Holographic Transformer processes different phase patterns.

Returns:
    Complex-valued tensor with shape [batch_size, seq_len, d_input] containing
    structured phase patterns for visualization purposes"""
    batch_size, seq_len, d_input = 1, 8, 4
    
    # Create data with specific phase patterns for visualization
    x = torch.zeros(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
    
    # Pattern 1: Gradual phase increase
    for t in range(seq_len):
        phase = (t / seq_len) * 2 * np.pi
        magnitude = 1.0 + 0.2 * np.sin(t)  # Varying magnitude
        x[0, t] = magnitude * torch.exp(1j * phase)
    
    return x


def print_attention_matrix(attn_matrix, title="Attention Matrix"):
    """Display attention weights in a human-readable tabular format.

Formats and prints attention matrices with appropriate row and column labels
for easy interpretation of attention patterns.

Args:
    attn_matrix: 2D tensor or numpy array containing attention weights
    title: Optional title for the printed matrix"""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    # Convert to numpy for easier printing
    if torch.is_tensor(attn_matrix):
        matrix = attn_matrix.cpu().numpy()
    else:
        matrix = attn_matrix
    
    seq_len = matrix.shape[-1]
    
    # Print column headers
    print("     ", end="")
    for j in range(seq_len):
        print(f"  {j:2d}  ", end="")
    print()
    
    # Print matrix with row labels
    for i in range(seq_len):
        print(f"{i:2d}: ", end="")
        for j in range(seq_len):
            print(f"{matrix[i, j]:5.3f}", end=" ")
        print()


def print_phase_matrix(phase_matrix, title="Phase Differences (radians)"):
    """Display phase differences between token pairs in a formatted table.

Formats and prints phase difference matrices with appropriate labels
for analyzing the relationship between phase differences and attention.

Args:
    phase_matrix: 2D tensor or numpy array containing phase differences
    title: Optional title for the printed matrix"""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    
    # Convert to numpy for easier printing
    if torch.is_tensor(phase_matrix):
        matrix = phase_matrix.cpu().numpy()
    else:
        matrix = phase_matrix
    
    seq_len = matrix.shape[-1]
    
    # Print column headers
    print("     ", end="")
    for j in range(seq_len):
        print(f"   {j:2d}   ", end="")
    print()
    
    # Print matrix with row labels
    for i in range(seq_len):
        print(f"{i:2d}: ", end="")
        for j in range(seq_len):
            print(f"{matrix[i, j]:6.3f}", end=" ")
        print()


def analyze_interference_patterns(attn, delta_phi):
    """Examine patterns of constructive and destructive interference in attention.

Calculates statistics on attention weights for positions with small phase
differences (constructive interference) versus large phase differences
(destructive interference), highlighting coherence decay effects.

Args:
    attn: Attention matrix tensor
    delta_phi: Phase difference matrix tensor"""
    print("\nInterference Pattern Analysis:")
    print("=" * 30)
    
    # Convert to numpy
    attn_np = attn.cpu().numpy()
    phase_np = delta_phi.cpu().numpy()
    
    # Find positions with small phase differences (constructive interference)
    small_phase_mask = np.abs(phase_np) < np.pi/4
    large_phase_mask = np.abs(phase_np) > 3*np.pi/4
    
    # Exclude diagonal (self-attention)
    seq_len = attn_np.shape[-1]
    non_diag_mask = ~np.eye(seq_len, dtype=bool)
    
    small_phase_attn = attn_np[small_phase_mask & non_diag_mask]
    large_phase_attn = attn_np[large_phase_mask & non_diag_mask]
    
    if len(small_phase_attn) > 0:
        print(f"Small phase differences (< π/4): {len(small_phase_attn)} pairs")
        print(f"  Average attention: {np.mean(small_phase_attn):.4f}")
        print(f"  Max attention: {np.max(small_phase_attn):.4f}")
    
    if len(large_phase_attn) > 0:
        print(f"Large phase differences (> 3π/4): {len(large_phase_attn)} pairs")
        print(f"  Average attention: {np.mean(large_phase_attn):.4f}")
        print(f"  Max attention: {np.max(large_phase_attn):.4f}")
    
    if len(small_phase_attn) > 0 and len(large_phase_attn) > 0:
        ratio = np.mean(small_phase_attn) / (np.mean(large_phase_attn) + 1e-8)
        print(f"Attention ratio (small/large phase): {ratio:.2f}")
        
        if ratio > 1.5:
            print("  → Strong coherence decay effect detected!")
        elif ratio > 1.1:
            print("  → Moderate coherence decay effect detected.")
        else:
            print("  → Weak coherence decay effect.")




def analyze_layer_differences(model, x):
    """Examine the evolution of attention patterns through model layers.

Runs the model and inspects attention patterns at each layer, calculating
statistics and highlighting how patterns transform through the network.

Args:
    model: Holographic Transformer model instance
    x: Input tensor for visualization"""
    print("\n" + "="*60)
    print("LAYER-WISE ATTENTION ANALYSIS")
    print("="*60)
    
    model.eval()
    num_layers = len(model.encoder.layers)
    
    with torch.no_grad():
        # Forward pass through encoder to get intermediate states
        x_proj = model.input_proj(x)
        
        # Add CLS token and positional encoding if used
        if model.encoder.use_cls_token:
            B = x_proj.size(0)
            cls_tokens = model.encoder.cls_token.expand(B, -1, -1)
            x_proj = torch.cat([cls_tokens, x_proj], dim=1)
        
        if model.encoder.use_positional_encoding:
            model.encoder._create_pos_encoding_if_needed(x_proj.device)
            seq_len = x_proj.size(1)
            if seq_len <= model.encoder.max_len:
                pos_enc = model.encoder.pos_encoding[:seq_len].unsqueeze(0)
                x_proj = x_proj + pos_enc
        
        x_proj = model.encoder.dropout(x_proj)
        
        # Pass through each layer and collect attention
        current_input = x_proj
        for layer_idx, layer in enumerate(model.encoder.layers):
            # Forward through this layer
            current_input = layer(current_input)
            
            # Get attention from this layer
            attn = layer.attention.get_last_attention()
            delta_phi = layer.attention.get_last_delta_phi()
            
            if attn is not None:
                print(f"\nLayer {layer_idx + 1}/{num_layers}:")
                print("-" * 30)
                
                # Use first batch, first head
                layer_attn = attn[0, 0]
                layer_delta_phi = delta_phi[0, 0] if delta_phi is not None else None
                
                # Attention statistics
                attn_entropy = -torch.sum(layer_attn * torch.log(layer_attn + 1e-8), dim=-1).mean().item()
                attn_max = torch.max(layer_attn).item()
                
                print(f"  Attention entropy: {attn_entropy:.3f}")
                print(f"  Max attention weight: {attn_max:.3f}")
                
                if layer_delta_phi is not None:
                    phase_std = torch.std(layer_delta_phi).item()
                    phase_range = torch.max(layer_delta_phi).item() - torch.min(layer_delta_phi).item()
                    print(f"  Phase difference std: {phase_std:.3f}")
                    print(f"  Phase difference range: {phase_range:.3f}")
                
                # Print compact attention matrix for deeper layers
                if layer_idx < 2:  # Full matrix for first 2 layers
                    print_attention_matrix(layer_attn, f"Layer {layer_idx + 1} Attention")
                else:  # Summary for deeper layers
                    print(f"  Attention matrix summary: shape {layer_attn.shape}")


def main():
    """Main function orchestrating the complete visualization pipeline.

Sets up the model, generates visualization data, and runs various
analysis functions to demonstrate the properties of the Holographic Transformer."""
    print("="*60)
    print("HOLOGRAPHIC TRANSFORMER ATTENTION VISUALIZATION")
    print("="*60)
    print(f"Device: {device}")
    print()
    
    # Model parameters
    d_input = 4
    d_model = 32
    n_heads = 2
    
    # Create model
    print("Creating HoloTransformer model...")
    model = HoloTransformer(
        d_input=d_input,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=64,
        num_layers=2,
        dropout=0.0,  # No dropout for clearer visualization
        use_cosine_sim=True,
        use_cls_token=True,
        task_type="classification",
        num_classes=2
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print()
    
    # Create visualization data
    print("Creating structured input data...")
    x = create_visualization_data()
    
    print("Input data shape:", x.shape)
    print("Input phases (first sample):")
    phases = torch.angle(x[0, :, 0]).cpu().numpy()
    magnitudes = torch.abs(x[0, :, 0]).cpu().numpy()
    
    for t in range(len(phases)):
        print(f"  Token {t}: magnitude={magnitudes[t]:.3f}, phase={phases[t]:.3f} ({phases[t]*180/np.pi:.1f}°)")
    
    print()
    
    
    # Analyze layer differences
    analyze_layer_differences(model, x)
    
    # Synthetic dataset comparison
    print("\n" + "="*60)
    print("SYNTHETIC DATASET ATTENTION PATTERNS")
    print("="*60)
    
    # Create dataset samples
    dataset = SyntheticComplexDataset(
        num_samples=4,
        seq_len=6,
        d_input=d_input,
        noise_level=0.05,
        device=device
    )
    
    for class_label in [0, 1]:
        # Find samples of each class
        class_samples = []
        for i, (sample, label) in enumerate(dataset):
            if label.item() == class_label and len(class_samples) < 1:
                class_samples.append(sample.unsqueeze(0))
        
        if class_samples:
            print(f"\nClass {class_label} Sample Analysis:")
            print("-" * 40)
            
            sample_x = class_samples[0]  # [1, T, d_input]
            
            with torch.no_grad():
                output = model(sample_x, return_attn=True)
                
                if 'attn' in output.aux:
                    attn = output.aux['attn'][0, 0]  # First head
                    delta_phi = output.aux['delta_phi'][0, 0]
                    
                    print_attention_matrix(attn, f"Class {class_label} Attention")
                    analyze_interference_patterns(attn, delta_phi)
                    
                    # Classification confidence
                    pred_probs = torch.softmax(output.outputs, dim=1)[0]
                    print(f"\nClassification confidence: [{pred_probs[0]:.3f}, {pred_probs[1]:.3f}]")
                    print(f"Predicted class: {torch.argmax(pred_probs).item()}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED!")
    print("="*60)
    print("\nKey Observations to Look For:")
    print("1. Coherence decay: Lower attention for larger phase differences")
    print("2. Constructive interference: Higher magnitudes for similar phases")
    print("3. Layer evolution: How patterns change through the network")
    print("4. Class sensitivity: Different patterns for different classes")


if __name__ == "__main__":
    main()
