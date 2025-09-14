"""
Testing padding mask functionality in Holographic Transformers.

This module verifies that padding masks are correctly implemented and that
attention toward padded positions is properly suppressed.
"""

import torch
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import HoloTransformer, HolographicMultiheadAttention
from holo_transformer.utils import get_device

device = get_device()


class TestPaddingMask:
    """Test class for padding mask functionality."""
    
    def test_attention_mask_effectiveness(self):
        """Verify that padding masks effectively suppress attention toward padded positions."""
        batch_size, seq_len, d_model = 2, 8, 32
        n_heads = 2
        
        attention = HolographicMultiheadAttention(d_model, n_heads).to(device)
        
        # Create input with some structure
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.cfloat).to(device)
        
        # Create padding mask - second half is padding
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, seq_len//2:] = True  # Second half is padding
        
        # Forward pass with mask
        output_masked = attention(x, padding_mask=padding_mask)
        attn_masked = attention.get_last_attention()  # [B, H, T, T]
        
        # Forward pass without mask
        output_unmasked = attention(x)
        attn_unmasked = attention.get_last_attention()
        
        # Check that attention to padded positions is near zero
        padded_attention = attn_masked[:, :, :, seq_len//2:]  # Attention to padded positions
        max_padded_attn = torch.max(padded_attention).item()
        
        print(f"Max attention to padded positions: {max_padded_attn:.6f}")
        assert max_padded_attn < 0.01, f"Attention to padded positions should be near zero, got {max_padded_attn}"
        
        # Check that attention weights sum to 1 for non-padded queries
        valid_queries = attn_masked[:, :, :seq_len//2, :]  # Attention from valid positions
        attn_sums = torch.sum(valid_queries, dim=-1)  # Sum over keys
        
        # Should sum to approximately 1
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
            "Attention weights should sum to 1 for valid queries"
    
    def test_encoder_with_mask(self):
        """Confirm that the encoder correctly processes and applies padding masks."""
        batch_size, seq_len, d_input = 2, 8, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            use_cls_token=False  # Easier to test without CLS
        ).to(device)
        
        # Input with padding in second half
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Mask second half as padding
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, seq_len//2:] = True  # Second half is padding
        
        # Forward with attention return
        output_masked = model(x, padding_mask=padding_mask, return_attn=True)
        output_unmasked = model(x, return_attn=True)
        
        # Get attention weights
        attn_masked = output_masked.aux['attn']  # [B, H, T, T]
        
        # Check that attention to padded positions is suppressed
        padded_attention = attn_masked[:, :, :, seq_len//2:]
        max_padded_attn = torch.max(padded_attention).item()
        
        print(f"Encoder max attention to padded positions: {max_padded_attn:.6f}")
        assert max_padded_attn < 0.01, "Encoder attention to padded positions should be near zero"
    
    def test_mask_with_cls_token(self):
        """Examine how padding masks interact with sequences containing CLS tokens."""
        batch_size, seq_len, d_input = 2, 6, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            use_cls_token=True
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Mask last two positions
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, -2:] = True
        
        output = model(x, padding_mask=padding_mask, return_attn=True)
        
        # Attention shape should be [B, H, T+1, T+1] (including CLS)
        attn = output.aux['attn']
        expected_seq_len = seq_len + 1  # +1 for CLS token
        
        assert attn.shape == (batch_size, 2, expected_seq_len, expected_seq_len)
        
        # Check CLS token can attend to all valid positions (not padded)
        cls_attention = attn[:, :, 0, :]  # CLS token's attention to all positions
        
        # CLS should attend to itself (position 0) and valid sequence positions (1 to seq_len-2)
        # but not to padded positions (seq_len-1 and seq_len)
        padded_positions_in_extended = [seq_len-1, seq_len]  # Last 2 positions in extended sequence
        cls_to_padded = cls_attention[:, :, padded_positions_in_extended]
        
        max_cls_to_padded = torch.max(cls_to_padded).item()
        print(f"CLS attention to padded positions: {max_cls_to_padded:.6f}")
        assert max_cls_to_padded < 0.01, "CLS token should not attend to padded positions"
    
    def test_reconstruction_with_mask(self):
        """Check that reconstruction loss calculations appropriately account for padding masks."""
        batch_size, seq_len, d_input = 2, 6, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1,
            lambda_recon=1.0,
            lambda_task=0.0  # Focus on reconstruction
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Create padding mask
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, -2:] = True  # Last 2 positions are padding
        
        # Forward pass
        output = model(x, padding_mask=padding_mask)
        
        # Loss should be computed
        assert 'recon' in output.loss_dict
        recon_loss = output.loss_dict['recon']
        
        assert not torch.isnan(recon_loss), "Reconstruction loss should not be NaN"
        assert recon_loss.item() >= 0, "Reconstruction loss should be non-negative"
        
        print(f"Reconstruction loss with mask: {recon_loss.item():.6f}")
    
    def test_different_mask_lengths(self):
        """Validate behavior when dealing with varying padding lengths within a batch."""
        batch_size, seq_len, d_input = 3, 8, 4
        d_model = 32
        
        model = HoloTransformer(
            d_input=d_input,
            d_model=d_model,
            n_heads=2,
            d_ff=64,
            num_layers=1
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_input, dtype=torch.cfloat).to(device)
        
        # Different padding for each sample in batch
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[0, -1:] = True    # Sample 0: 1 padded position
        padding_mask[1, -3:] = True    # Sample 1: 3 padded positions  
        padding_mask[2, -2:] = True    # Sample 2: 2 padded positions
        
        output = model(x, padding_mask=padding_mask, return_attn=True)
        
        attn = output.aux['attn']
        
        # Check each sample has different attention patterns due to different padding
        for sample_idx in range(batch_size):
            sample_attn = attn[sample_idx]  # [H, T+1, T+1] (including CLS token)
            
            # Find padded positions for this sample (adjust for CLS token)
            padded_positions = torch.where(padding_mask[sample_idx])[0] + 1  # +1 for CLS token
            
            if len(padded_positions) > 0:
                # Attention to padded positions should be near zero
                attn_to_padded = sample_attn[:, :, padded_positions]
                max_attn_to_padded = torch.max(attn_to_padded).item()
                
                print(f"Sample {sample_idx} max attention to padded positions: {max_attn_to_padded:.6f}")
                assert max_attn_to_padded < 0.01, \
                    f"Sample {sample_idx} attention to padded positions should be near zero"


def run_mask_tests():
    """Run all padding mask tests."""
    print("Running padding mask tests...")
    
    test_class = TestPaddingMask()
    
    test_methods = [
        test_class.test_attention_mask_effectiveness,
        test_class.test_encoder_with_mask,
        test_class.test_mask_with_cls_token,
        test_class.test_reconstruction_with_mask,
        test_class.test_different_mask_lengths,
    ]
    
    for i, test_method in enumerate(test_methods, 1):
        try:
            test_method()
            print(f"✓ Test {i}/{len(test_methods)}: {test_method.__name__}")
        except Exception as e:
            print(f"✗ Test {i}/{len(test_methods)}: {test_method.__name__} - {e}")
            raise
    
    print("All padding mask tests passed! ✓")


if __name__ == "__main__":
    run_mask_tests()
