"""
Minimal example demonstrating Holographic Transformer training on synthetic data.

This script provides a complete walkthrough of training a Holographic Transformer
on a synthetic complex-valued dataset for binary classification.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to import holo_transformer
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from holo_transformer import HoloTransformer, set_seed, get_gradient_norm, SyntheticComplexDataset
from holo_transformer.utils import get_device

# Set device and seed
device = get_device()
# set_seed(42)

def main():
    """Main function containing the complete training and evaluation pipeline."""
    print("="*60)
    print("HOLOGRAPHIC TRANSFORMER MINIMAL TRAINING EXAMPLE")
    print("="*60)
    print(f"Device: {device}")
    print()
    
    # Hyperparameters
    batch_size = 8
    seq_len = 16
    d_input = 8
    d_model = 64
    n_heads = 4
    num_epochs = 5
    lr = 1e-3
    
    print("Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dimension: {d_input}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    train_dataset = SyntheticComplexDataset(
        num_samples=200,
        seq_len=seq_len,
        d_input=d_input,
        noise_level=0.1,
        device=device
    )
    
    val_dataset = SyntheticComplexDataset(
        num_samples=50,
        seq_len=seq_len,
        d_input=d_input,
        noise_level=0.1,
        device=device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Creating HoloTransformer model...")
    model = HoloTransformer(
        d_input=d_input,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        num_layers=3,
        dropout=0.1,
        use_cosine_sim=True,
        use_cls_token=True,
        lambda_recon=0.5,
        lambda_task=1.0,
        task_type="classification",
        num_classes=2
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    # Training loop - iterates over epochs and performs training and validation
    print("Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_task_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            output = model(x, y=y, return_attn=(batch_idx == 0 and epoch == 0))
            
            loss = output.loss_dict['total']
            recon_loss = output.loss_dict['recon']
            task_loss = output.loss_dict['task']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_task_loss += task_loss.item()
            
            # Accuracy
            pred = torch.argmax(output.outputs, dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            
            # Print attention info for first batch of first epoch
            if batch_idx == 0 and epoch == 0:
                print("Sample attention and phase information:")
                if 'attn' in output.aux:
                    attn = output.aux['attn']
                    delta_phi = output.aux['delta_phi']
                    print(f"  Attention shape: {attn.shape}")
                    print(f"  Phase difference shape: {delta_phi.shape}")
                    print(f"  Max attention weight: {torch.max(attn):.4f}")
                    print(f"  Phase difference range: [{torch.min(delta_phi):.3f}, {torch.max(delta_phi):.3f}]")
                print()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x, y=y)
                loss = output.loss_dict['total']
                
                val_loss += loss.item()
                
                pred = torch.argmax(output.outputs, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        
        # Epoch statistics
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_task_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        grad_norm = get_gradient_norm(model)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, Task: {train_task_loss:.4f})")
        print(f"  Train - Accuracy: {train_acc:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.3f}")
        print(f"  Grad norm: {grad_norm:.4f}, LR: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ New best validation accuracy: {best_val_acc:.3f}")
        
        print()
        
        # Step scheduler
        scheduler.step()
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print()
    
    # Demonstration of inference and analysis
    print("="*60)
    print("INFERENCE AND ANALYSIS DEMONSTRATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Get one batch for analysis
        x_test, y_test = next(iter(val_loader))
        
        # Forward pass with attention return
        output = model(x_test, return_attn=True)
        
        pred = torch.argmax(output.outputs, dim=1)
        pred_probs = torch.softmax(output.outputs, dim=1)
        
        print("Predictions vs Ground Truth:")
        for i in range(min(5, len(y_test))):
            print(f"  Sample {i}: Pred={pred[i].item()}, True={y_test[i].item()}, "
                  f"Prob=[{pred_probs[i, 0]:.3f}, {pred_probs[i, 1]:.3f}]")
        
        print(f"\nOverall test accuracy: {torch.mean((pred == y_test).float()).item():.3f}")
        
        # Reconstruction quality
        recon_mse = torch.mean(torch.abs(output.x_hat - x_test)**2).item()
        print(f"Reconstruction MSE: {recon_mse:.6f}")
        
        # Attention analysis
        if 'attn' in output.aux:
            attn = output.aux['attn']  # [B, H, T, T]
            delta_phi = output.aux['delta_phi']
            
            print(f"\nAttention Analysis:")
            print(f"  Attention entropy (avg): {-torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean():.3f}")
            print(f"  Phase difference std: {torch.std(delta_phi).item():.3f}")
            
            # Analyze coherence patterns
            coherence_scores = torch.exp(-torch.abs(delta_phi))
            avg_coherence = torch.mean(coherence_scores).item()
            print(f"  Average coherence score: {avg_coherence:.3f}")
    
    print()
    
    print("="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
