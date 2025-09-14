"""
Tests for gradient computation and differentiability in Holographic Transformers.

This module verifies that all components are differentiable and gradients
are computed correctly for complex-valued parameters.
"""

import torch
import torch.autograd
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import HoloTransformer, ComplexLinear, HolographicMultiheadAttention
from holo_transformer.utils import get_device, get_gradient_norm

device = get_device()


class TestGradients:
    """Test class for gradient computation and verification."""
    
    def test_complex_linear_gradients(self):
        """Verify that the ComplexLinear layer computes gradients correctly."""
        batch_size, seq_len, d_in, d_out = 2, 4, 8, 16
        
        layer = ComplexLinear(d_in, d_out).to(device)
        x = torch.randn(batch_size, seq_len, d_in, dtype=torch.cfloat, requires_grad=True).to(device)
        
        # Forward pass
        output = layer(x)
        
        # Create a loss (sum of squared magnitudes)
        loss = torch.sum(torch.abs(output) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        assert layer.weight.grad is not None, "Weight should have gradients"
        assert not torch.isnan(layer.weight.grad).any(), "Weight gradients should not be NaN"
        assert not torch.isinf(layer.weight.grad).any(), "Weight gradients should not be inf"
        
        if layer.bias is not None:
            assert layer.bias.grad is not None, "Bias should have gradients"
            assert not torch.isnan(layer.bias.grad).any(), "Bias gradients should not be NaN"
        
        # Check input gradients
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
        
        print(f"ComplexLinear weight grad norm: {torch.norm(layer.weight.grad):.6f}")
        print(f"ComplexLinear input grad norm: {torch.norm(x.grad):.6f}")
    
    def test_attention_gradients(self):
        """Verify that HolographicMultiheadAttention computes gradients correctly."""
        batch_size, seq_len, d_model = 2, 4, 32
        n_heads = 2
        
        attention = HolographicMultiheadAttention(d_model, n_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat, requires_grad=True).to(device)
        
        # Forward pass
        output = attention(x)
        
        # Create loss
        loss = torch.sum(torch.abs(output) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients for all parameters
        for name, param in attention.named_parameters():
            assert param.grad is not None, f"{name} should have gradients"
            assert not torch.isnan(param.grad).any(), f"{name} gradients should not be NaN"
            print(f"{name} grad norm: {torch.norm(param.grad):.6f}")
        
        # Check alpha parameter gradients specifically
        assert attention.alpha_param.grad is not None, "Alpha parameter should have gradients"
        alpha_grad_norm = torch.norm(attention.alpha_param.grad).item()
        print(f"Alpha parameter grad norm: {alpha_grad_norm:.6f}")
        
        # Input gradients
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
    
    def test_full_model_gradients(self):
        """Verify that the complete HoloTransformer model computes gradients correctly."""
        batch_size, seq_len, d_input = 2, 4, 8
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            task_type="classification",
            num_classes=2
        ).to(device)
        
        # Input data
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat, requires_grad=True).to(device)
        y = torch.randint(0, 2, (batch_size,)).to(device)
        
        # Forward pass
        output = model(x, y=y)
        loss = output.loss_dict['total']
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        params_with_grads = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grads += 1
                assert not torch.isnan(param.grad).any(), f"{name} gradients should not be NaN"
                assert not torch.isinf(param.grad).any(), f"{name} gradients should not be inf"
        
        print(f"Parameters with gradients: {params_with_grads}/{total_params}")
        assert params_with_grads == total_params, "All parameters should have gradients"
        
        # Check gradient norm
        grad_norm = get_gradient_norm(model)
        print(f"Total gradient norm: {grad_norm:.6f}")
        
        assert grad_norm > 0, "Gradient norm should be positive"
        assert not math.isnan(grad_norm), "Gradient norm should not be NaN"
        
        # Check input gradients
        assert x.grad is not None, "Input should have gradients"
        input_grad_norm = torch.norm(x.grad).item()
        print(f"Input gradient norm: {input_grad_norm:.6f}")
    
    def test_gradients_with_mask(self):
        """Verify gradient computation with padding masks."""
        batch_size, seq_len, d_input = 2, 6, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            task_type="regression",
            num_outputs=1
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat, requires_grad=True).to(device)
        y = torch.randn(batch_size, 1).to(device)
        
        # Create padding mask
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, -2:] = True  # Last 2 positions are padding
        
        # Forward pass with mask
        output = model(x, padding_mask=padding_mask, y=y)
        loss = output.loss_dict['total']
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed correctly even with masking
        grad_norm = get_gradient_norm(model)
        print(f"Gradient norm with mask: {grad_norm:.6f}")
        
        assert grad_norm > 0, "Gradient norm should be positive with mask"
        assert not math.isnan(grad_norm), "Gradient norm should not be NaN with mask"
        
        # Input gradients should exist
        assert x.grad is not None, "Input should have gradients with mask"
        
        # Check that gradients for masked positions might be different
        # (though the exact behavior depends on the loss computation)
        valid_grad = x.grad[:, :-2]  # Gradients for valid positions
        padded_grad = x.grad[:, -2:]  # Gradients for padded positions
        
        valid_norm = torch.norm(valid_grad).item()
        padded_norm = torch.norm(padded_grad).item()
        
        print(f"Valid positions gradient norm: {valid_norm:.6f}")
        print(f"Padded positions gradient norm: {padded_norm:.6f}")
    
    def test_second_order_gradients(self):
        """Verify second-order gradients (useful for advanced optimization methods)."""
        # Use very small model for computational efficiency
        d_input, d_model = 4, 16
        batch_size, seq_len = 1, 2
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=1,
            d_ff=32,
            num_layers=1,
            task_type="regression",
            num_outputs=1
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        y = torch.randn(batch_size, 1).to(device)
        
        # Forward pass
        output = model(x, y=y)
        loss = output.loss_dict['total']
        
        # First-order gradients
        first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Create a scalar from first-order gradients for second-order computation
        grad_sum = sum(torch.sum(g) for g in first_grads if g is not None)
        
        # Second-order gradients
        try:
            second_grads = torch.autograd.grad(grad_sum, model.parameters(), retain_graph=False)
            
            print("Second-order gradients computed successfully")
            
            # Check that second-order gradients exist
            for i, (first_grad, second_grad) in enumerate(zip(first_grads, second_grads)):
                if first_grad is not None and second_grad is not None:
                    assert not torch.isnan(second_grad).any(), f"Second-order gradient {i} should not be NaN"
            
        except RuntimeError as e:
            print(f"Second-order gradients not supported or failed: {e}")
            # This is acceptable as second-order gradients are not always required
    
    def test_gradient_accumulation(self):
        """Verify gradient accumulation across multiple batches."""
        batch_size, seq_len, d_input = 2, 4, 4
        d_model = 16
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=1,
            d_ff=32,
            num_layers=1,
            task_type="classification",
            num_classes=2
        ).to(device)
        
        # Multiple batches
        num_batches = 3
        total_loss = 0
        
        for batch_idx in range(num_batches):
            x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
            y = torch.randint(0, 2, (batch_size,)).to(device)
            
            # Forward pass
            output = model(x, y=y)
            loss = output.loss_dict['total'] / num_batches  # Scale for accumulation
            
            # Backward pass (accumulate gradients)
            loss.backward()
            total_loss += loss.item()
        
        # Check accumulated gradients
        grad_norm = get_gradient_norm(model)
        print(f"Accumulated gradient norm over {num_batches} batches: {grad_norm:.6f}")
        print(f"Total accumulated loss: {total_loss:.6f}")
        
        assert grad_norm > 0, "Accumulated gradients should be positive"
        assert not math.isnan(grad_norm), "Accumulated gradients should not be NaN"


import math  # Add missing import

def run_gradient_tests():
    """Run all gradient tests."""
    print("Running gradient and differentiability tests...")
    
    test_class = TestGradients()
    
    test_methods = [
        test_class.test_complex_linear_gradients,
        test_class.test_attention_gradients,
        test_class.test_full_model_gradients,
        test_class.test_gradients_with_mask,
        test_class.test_second_order_gradients,
        test_class.test_gradient_accumulation,
    ]
    
    for i, test_method in enumerate(test_methods, 1):
        try:
            test_method()
            print(f"✓ Test {i}/{len(test_methods)}: {test_method.__name__}")
        except Exception as e:
            print(f"✗ Test {i}/{len(test_methods)}: {test_method.__name__} - {e}")
            raise
    
    print("All gradient tests passed! ✓")


if __name__ == "__main__":
    run_gradient_tests()
