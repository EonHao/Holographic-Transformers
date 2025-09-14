"""
Tests for validating output shapes and data types in Holographic Transformers.

This module verifies that all components produce the correct output shapes and
data types when processing complex-valued tensors.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import (
    ComplexLinear, ComplexLayerNorm, ComplexFFN, HolographicMultiheadAttention,
    HoloTransformerEncoder, DualHeadDecoder, HoloTransformer,
    get_complex_positional_encoding
)
from holo_transformer.utils import get_device

device = get_device()


class TestShapesAndDTypes:
    """Test class for shapes and data types."""
    
    def test_complex_linear_shapes(self):
        """Verify output shapes and data types of the ComplexLinear layer."""
        batch_size, seq_len, d_in, d_out = 4, 16, 32, 64
        
        layer = ComplexLinear(d_in, d_out).to(device)
        x = torch.randn(batch_size, seq_len, d_in, dtype=torch.cfloat).to(device)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_out)
        assert output.dtype == torch.cfloat
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_complex_layer_norm_shapes(self):
        """Verify output shapes and data types of the ComplexLayerNorm."""
        batch_size, seq_len, d_model = 4, 16, 64
        
        # Test with separate affine parameters
        layer1 = ComplexLayerNorm(d_model, shared_affine=False).to(device)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        output1 = layer1(x)
        
        assert output1.shape == (batch_size, seq_len, d_model)
        assert output1.dtype == torch.cfloat
        
        # Test with shared affine parameters
        layer2 = ComplexLayerNorm(d_model, shared_affine=True).to(device)
        output2 = layer2(x)
        
        assert output2.shape == (batch_size, seq_len, d_model)
        assert output2.dtype == torch.cfloat
    
    def test_complex_ffn_shapes(self):
        """Verify output shapes and data types of the ComplexFFN."""
        batch_size, seq_len, d_model, d_ff = 4, 16, 64, 128
        
        # Test GELU activation
        ffn_gelu = ComplexFFN(d_model, d_ff, activation="gelu").to(device)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        output_gelu = ffn_gelu(x)
        
        assert output_gelu.shape == (batch_size, seq_len, d_model)
        assert output_gelu.dtype == torch.cfloat
        
        # Test ModReLU activation
        ffn_modrelu = ComplexFFN(d_model, d_ff, activation="modrelu").to(device)
        output_modrelu = ffn_modrelu(x)
        
        assert output_modrelu.shape == (batch_size, seq_len, d_model)
        assert output_modrelu.dtype == torch.cfloat
    
    def test_holographic_attention_shapes(self):
        """Verify output shapes and data types of the HolographicMultiheadAttention."""
        batch_size, seq_len, d_model, n_heads = 4, 16, 64, 4
        
        attention = HolographicMultiheadAttention(d_model, n_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.cfloat
        
        # Check that attention weights and phase differences are stored
        attn_weights = attention.get_last_attention()
        delta_phi = attention.get_last_delta_phi()
        
        assert attn_weights is not None
        assert delta_phi is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        assert delta_phi.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_encoder_shapes(self):
        """Test HoloTransformerEncoder shapes and dtypes."""
        batch_size, seq_len, d_model = 4, 16, 64
        n_heads, d_ff, num_layers = 4, 128, 2
        
        # Test with CLS token
        encoder_with_cls = HoloTransformerEncoder(
            d_model, n_heads, d_ff, num_layers, use_cls_token=True
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        output_with_cls = encoder_with_cls(x)
        
        assert output_with_cls.shape == (batch_size, seq_len + 1, d_model)  # +1 for CLS
        assert output_with_cls.dtype == torch.cfloat
        
        # Test without CLS token
        encoder_no_cls = HoloTransformerEncoder(
            d_model, n_heads, d_ff, num_layers, use_cls_token=False
        ).to(device)
        
        output_no_cls = encoder_no_cls(x)
        
        assert output_no_cls.shape == (batch_size, seq_len, d_model)
        assert output_no_cls.dtype == torch.cfloat
    
    def test_decoder_shapes(self):
        """Test DualHeadDecoder shapes and dtypes."""
        batch_size, seq_len, d_model, d_input = 4, 16, 64, 32
        num_classes = 3
        
        # Test classification with CLS token
        decoder_cls = DualHeadDecoder(
            d_model, d_input, task_type="classification", 
            num_classes=num_classes, use_cls_token=True
        ).to(device)
        
        # Input has CLS token
        encoded = torch.randn(batch_size, seq_len + 1, d_model, dtype=torch.cfloat).to(device)
        x_hat, outputs = decoder_cls(encoded)
        
        assert x_hat.shape == (batch_size, seq_len, d_input)
        assert x_hat.dtype == torch.cfloat
        assert outputs.shape == (batch_size, num_classes)
        assert outputs.dtype == torch.float32
        
        # Test regression without CLS token
        decoder_reg = DualHeadDecoder(
            d_model, d_input, task_type="regression",
            num_outputs=2, use_cls_token=False
        ).to(device)
        
        encoded_no_cls = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        x_hat_reg, outputs_reg = decoder_reg(encoded_no_cls)
        
        assert x_hat_reg.shape == (batch_size, seq_len, d_input)
        assert x_hat_reg.dtype == torch.cfloat
        assert outputs_reg.shape == (batch_size, 2)
        assert outputs_reg.dtype == torch.float32
    
    def test_full_model_shapes(self):
        """Test complete HoloTransformer model shapes and dtypes."""
        batch_size, seq_len, d_input = 4, 16, 8
        d_model, n_heads, d_ff, num_layers = 64, 4, 128, 2
        num_classes = 2
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            task_type="classification",
            num_classes=num_classes
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # Forward pass without labels
        output_no_labels = model(x)
        
        assert output_no_labels.x_hat.shape == (batch_size, seq_len, d_input)
        assert output_no_labels.x_hat.dtype == torch.cfloat
        assert output_no_labels.outputs.shape == (batch_size, num_classes)
        assert output_no_labels.outputs.dtype == torch.float32
        assert 'recon' in output_no_labels.loss_dict
        
        # Forward pass with labels
        output_with_labels = model(x, y=y)
        
        assert 'recon' in output_with_labels.loss_dict
        assert 'task' in output_with_labels.loss_dict
        assert 'total' in output_with_labels.loss_dict
        
        # Forward pass with attention return
        output_with_attn = model(x, return_attn=True)
        
        assert 'attn' in output_with_attn.aux
        assert 'delta_phi' in output_with_attn.aux
    
    def test_positional_encoding_shapes(self):
        """Test complex positional encoding shapes and dtypes."""
        max_len, d_model = 100, 64
        
        pos_enc = get_complex_positional_encoding(max_len, d_model, device)
        
        assert pos_enc.shape == (max_len, d_model)
        assert pos_enc.dtype == torch.cfloat
        assert not torch.isnan(pos_enc).any()
        assert not torch.isinf(pos_enc).any()


def run_shape_tests():
    """Run all shape tests."""
    print("Running shape and dtype tests...")
    
    test_class = TestShapesAndDTypes()
    
    test_methods = [
        test_class.test_complex_linear_shapes,
        test_class.test_complex_layer_norm_shapes,
        test_class.test_complex_ffn_shapes,
        test_class.test_holographic_attention_shapes,
        test_class.test_encoder_shapes,
        test_class.test_decoder_shapes,
        test_class.test_full_model_shapes,
        test_class.test_positional_encoding_shapes,
    ]
    
    for i, test_method in enumerate(test_methods, 1):
        try:
            test_method()
            print(f"✓ Test {i}/{len(test_methods)}: {test_method.__name__}")
        except Exception as e:
            print(f"✗ Test {i}/{len(test_methods)}: {test_method.__name__} - {e}")
            raise
    
    print("All shape tests passed! ✓")


if __name__ == "__main__":
    run_shape_tests()
