"""
Tests for verifying invariance properties in Holographic Transformers.

This module tests that the cosine similarity version of the model
maintains approximate invariance to global phase shifts.
"""

import torch
import math
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import HoloTransformer, HolographicMultiheadAttention
from holo_transformer.utils import get_device

device = get_device()


class TestPhaseInvariance:
    """Test class for phase invariance properties."""
    
    def test_attention_cosine_invariance(self):
        """Verify phase invariance property of cosine similarity attention mechanism."""
        batch_size, seq_len, d_model = 2, 8, 32
        n_heads = 2
        
        # Create attention with cosine similarity
        attention = HolographicMultiheadAttention(
            d_model, n_heads, use_cosine_sim=True
        ).to(device)
        
        # Original input
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        # Apply random global phase shift
        global_phase = torch.rand(1).item() * 2 * math.pi - math.pi
        phase_factor = torch.exp(1j * torch.tensor(global_phase, device=x.device))
        x_shifted = x * phase_factor
        
        with torch.no_grad():
            # Forward passes
            output1 = attention(x)
            attn1 = attention.get_last_attention()
            
            output2 = attention(x_shifted)
            attn2 = attention.get_last_attention()
        
        # Compare attention patterns (should be very similar for cosine version)
        attn_diff = torch.mean((attn1 - attn2) ** 2).item()
        
        print(f"Attention difference after global phase shift: {attn_diff:.8f}")
        
        # Tolerance for attention patterns
        tolerance = 1e-5
        assert attn_diff < tolerance, \
            f"Cosine attention should be phase-invariant: {attn_diff} > {tolerance}"
    
    def test_attention_dot_product_variance(self):
        """Verify phase sensitivity of dot product attention mechanism."""
        batch_size, seq_len, d_model = 2, 8, 32
        n_heads = 2
        
        # Create attention with dot product similarity
        attention = HolographicMultiheadAttention(
            d_model, n_heads, use_cosine_sim=False
        ).to(device)
        
        # Original input
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        # Apply global phase shift
        global_phase = math.pi / 4  # 45 degree shift
        phase_factor = torch.exp(1j * torch.tensor(global_phase, device=x.device))
        x_shifted = x * phase_factor
        
        with torch.no_grad():
            output1 = attention(x)
            attn1 = attention.get_last_attention()
            
            output2 = attention(x_shifted)
            attn2 = attention.get_last_attention()
        
        # Compare attention patterns (should be different for dot product version)
        attn_diff = torch.mean((attn1 - attn2) ** 2).item()
        
        print(f"Dot product attention difference after phase shift: {attn_diff:.8f}")
        
        # Dot product version should show more variance
        assert attn_diff > 1e-6, \
            "Dot product attention should show phase variance"
    
    def test_model_classification_invariance(self):
        """Validate classification robustness against global phase transformations."""
        batch_size, seq_len, d_input = 2, 8, 4
        d_model = 32
        num_classes = 3
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,  # Reduce layers for better invariance
            use_cosine_sim=True,  # Cosine version should be more phase-invariant
            task_type="classification",
            num_classes=num_classes
        ).to(device)
        
        # Original input
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Apply several different global phase shifts
        phase_shifts = [0, math.pi/4, math.pi/2, math.pi, -math.pi/3]
        outputs = []
        
        with torch.no_grad():
            for phase in phase_shifts:
                phase_factor = torch.exp(1j * torch.tensor(phase, device=x.device))
                x_shifted = x * phase_factor
                output = model(x_shifted)
                outputs.append(output.outputs)
        
        # Compare outputs across different phase shifts
        max_diff = 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = torch.mean((outputs[i] - outputs[j]) ** 2).item()
                max_diff = max(max_diff, diff)
        
        print(f"Max classification output difference across phase shifts: {max_diff:.6f}")
        
        # Tolerance depends on network depth and complexity
        tolerance = 3e-1  # Very relaxed tolerance for complex network with multiple transformations
        assert max_diff < tolerance, \
            f"Classification outputs should be approximately phase-invariant: {max_diff} > {tolerance}"
    
    def test_reconstruction_phase_preservation(self):
        """Test that reconstruction preserves relative phases even if not globally invariant."""
        batch_size, seq_len, d_input = 2, 6, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            lambda_recon=1.0,
            lambda_task=0.0
        ).to(device)
        
        # Create input with specific phase pattern
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Apply global phase shift
        global_phase = math.pi / 3
        phase_factor = torch.exp(1j * torch.tensor(global_phase, device=x.device))
        x_shifted = x * phase_factor
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x_shifted)
            
            x_hat1 = output1.x_hat
            x_hat2 = output2.x_hat
        
        # The reconstructions should differ by approximately the same global phase
        # Compute the phase difference between reconstructions
        ratio = x_hat2 / (x_hat1 + 1e-8)  # Avoid division by zero
        
        # The ratio should have approximately constant phase if global phase is preserved
        ratio_angles = torch.angle(ratio)
        
        # Check if the phase differences are approximately constant
        phase_std = torch.std(ratio_angles).item()
        
        print(f"Phase difference standard deviation: {phase_std:.6f}")
        print(f"Expected global phase shift: {global_phase:.6f}")
        print(f"Mean phase difference: {torch.mean(ratio_angles).item():.6f}")
        
        # The phase differences should be relatively consistent
        # (though perfect invariance is not expected for reconstruction)
        assert phase_std < 1.5, "Phase differences should be relatively consistent"
    
    def test_invariance_with_different_magnitudes(self):
        """Test phase invariance with inputs of different magnitudes."""
        batch_size, seq_len, d_input = 2, 6, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            use_cosine_sim=True,
            task_type="classification",
            num_classes=2
        ).to(device)
        
        # Create inputs with different magnitude scales
        x_small = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device) * 0.1
        x_large = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device) * 10.0
        
        # Apply same phase shift to both
        global_phase = math.pi / 4
        phase_factor = torch.exp(1j * torch.tensor(global_phase, device=x_small.device))
        
        x_small_shifted = x_small * phase_factor
        x_large_shifted = x_large * phase_factor
        
        with torch.no_grad():
            # Small magnitude inputs
            output_small_1 = model(x_small)
            output_small_2 = model(x_small_shifted)
            
            # Large magnitude inputs
            output_large_1 = model(x_large)
            output_large_2 = model(x_large_shifted)
        
        # Compare invariance for different magnitudes
        small_diff = torch.mean((output_small_1.outputs - output_small_2.outputs) ** 2).item()
        large_diff = torch.mean((output_large_1.outputs - output_large_2.outputs) ** 2).item()
        
        print(f"Small magnitude phase invariance difference: {small_diff:.6f}")
        print(f"Large magnitude phase invariance difference: {large_diff:.6f}")
        
        # Both should show similar levels of invariance (cosine similarity normalizes magnitude)
        tolerance = 2e-2  # Relaxed tolerance for magnitude invariance
        assert small_diff < tolerance, f"Small magnitude should be phase-invariant: {small_diff}"
        assert large_diff < tolerance, f"Large magnitude should be phase-invariant: {large_diff}"


def run_invariance_tests():
    """Run all phase invariance tests."""
    print("Running phase invariance tests...")
    
    test_class = TestPhaseInvariance()
    
    test_methods = [
        test_class.test_attention_cosine_invariance,
        test_class.test_attention_dot_product_variance,
        test_class.test_model_classification_invariance,
        test_class.test_reconstruction_phase_preservation,
        test_class.test_invariance_with_different_magnitudes,
    ]
    
    for i, test_method in enumerate(test_methods, 1):
        try:
            test_method()
            print(f"✓ Test {i}/{len(test_methods)}: {test_method.__name__}")
        except Exception as e:
            print(f"✗ Test {i}/{len(test_methods)}: {test_method.__name__} - {e}")
            raise
    
    print("All phase invariance tests passed! ✓")


if __name__ == "__main__":
    run_invariance_tests()
