#!/usr/bin/env python
"""
Test script for new features from old project:
1. Combined Loss (MSE + Peak)
2. MC Dropout Uncertainty
3. Per-layer Dropout
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


def test_mse_peak_loss():
    """Test MSEWithPeakLoss implementation."""
    print("\n" + "="*60)
    print("Testing MSEWithPeakLoss...")
    print("="*60)
    
    from training.losses import MSEWithPeakLoss
    
    # Create loss function
    loss_fn = MSEWithPeakLoss(mse_weight=0.7, peak_weight=0.3)
    
    # Create fake heatmaps
    B, K, H, W = 2, 4, 64, 64
    
    # Target: peak at (20, 30)
    target = torch.zeros(B, K, H, W)
    for b in range(B):
        for k in range(K):
            target[b, k, 20, 30] = 10.0  # Peak at y=20, x=30
    
    # Prediction 1: same peak location (should have low loss)
    pred_good = torch.zeros(B, K, H, W)
    for b in range(B):
        for k in range(K):
            pred_good[b, k, 20, 30] = 10.0
    
    # Prediction 2: offset peak (should have higher loss)
    pred_offset = torch.zeros(B, K, H, W)
    for b in range(B):
        for k in range(K):
            pred_offset[b, k, 25, 35] = 10.0  # 5px offset in y, 5px in x
    
    # Calculate losses
    loss_good = loss_fn(pred_good, target)
    loss_offset = loss_fn(pred_offset, target)
    
    print(f"Loss (perfect prediction): {loss_good.item():.6f}")
    print(f"Loss (offset prediction):  {loss_offset.item():.6f}")
    
    assert loss_offset > loss_good, "Offset prediction should have higher loss"
    print("✓ MSEWithPeakLoss test passed!")
    
    return True


def test_per_layer_dropout():
    """Test per-layer dropout in UNet."""
    print("\n" + "="*60)
    print("Testing Per-layer Dropout...")
    print("="*60)
    
    from models.unet import UNet
    
    # Create model with per-layer dropout
    down_dropout = [0.1, 0.2, 0.3, 0.4]
    up_dropout = [0.3, 0.2, 0.1, 0.0]
    
    model = UNet(
        in_channels=3,
        num_keypoints=4,
        down_dropout=down_dropout,
        up_dropout=up_dropout
    )
    
    # Verify dropout layers exist
    print(f"Encoder dropout rates: {down_dropout}")
    print(f"Decoder dropout rates: {up_dropout}")
    
    # Check that down blocks have dropout
    assert model.down1.dropout is not None, "down1 should have dropout"
    assert model.down2.dropout is not None, "down2 should have dropout"
    assert model.down3.dropout is not None, "down3 should have dropout"
    assert model.down4.dropout is not None, "down4 should have dropout"
    
    # Check up blocks with dropout
    assert model.up1.dropout is not None, "up1 should have dropout"
    assert model.up2.dropout is not None, "up2 should have dropout"
    assert model.up3.dropout is not None, "up3 should have dropout"
    assert model.up4.dropout is None, "up4 should NOT have dropout (rate=0)"
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    model.train()  # Enable dropout
    y_train = model(x)
    model.eval()   # Disable dropout
    y_eval = model(x)
    
    print(f"Output shape: {y_train.shape}")
    assert y_train.shape == (1, 4, 256, 256), "Wrong output shape"
    
    print("✓ Per-layer dropout test passed!")
    
    return True


def test_mc_dropout_uncertainty():
    """Test Monte Carlo dropout for uncertainty estimation."""
    print("\n" + "="*60)
    print("Testing MC Dropout Uncertainty...")
    print("="*60)
    
    from models.unet import UNet
    
    # Create model with dropout
    model = UNet(
        in_channels=3,
        num_keypoints=4,
        down_dropout=[0.3, 0.3, 0.3, 0.3],
        up_dropout=[0.3, 0.3, 0.3, 0.3]
    )
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 128, 128)
    
    # Get predictions with uncertainty
    n_samples = 5
    mean_heatmaps, std_heatmaps, all_samples = model.predict_with_uncertainty(x, n_samples=n_samples)
    
    print(f"Input shape: {x.shape}")
    print(f"Mean heatmaps shape: {mean_heatmaps.shape}")
    print(f"Std (uncertainty) shape: {std_heatmaps.shape}")
    print(f"All samples shape: {all_samples.shape}")
    
    # Verify shapes
    assert mean_heatmaps.shape == (2, 4, 128, 128), "Wrong mean shape"
    assert std_heatmaps.shape == (2, 4, 128, 128), "Wrong std shape"
    assert all_samples.shape == (n_samples, 2, 4, 128, 128), "Wrong samples shape"
    
    # Verify std is non-zero (dropout should create variation)
    assert std_heatmaps.mean() > 0, "Std should be > 0 with dropout"
    print(f"Mean uncertainty: {std_heatmaps.mean().item():.6f}")
    
    # Verify model is back in eval mode
    assert not model.training, "Model should be in eval mode after predict_with_uncertainty"
    
    print("✓ MC Dropout Uncertainty test passed!")
    
    return True


def test_backwards_compatibility():
    """Test that old code still works (backwards compatibility)."""
    print("\n" + "="*60)
    print("Testing Backwards Compatibility...")
    print("="*60)
    
    from models.unet import UNet
    
    # Old way: single dropout_rate
    model_old_style = UNet(
        in_channels=3,
        num_keypoints=4,
        dropout_rate=0.3
    )
    
    # Check that bottleneck has dropout but other layers don't
    # With old style, down_dropout defaults to [0,0,0,dropout_rate]
    assert model_old_style.down4.dropout is not None, "Bottleneck should have dropout"
    
    x = torch.randn(1, 3, 256, 256)
    y = model_old_style(x)
    
    assert y.shape == (1, 4, 256, 256), "Wrong output shape"
    
    print("✓ Backwards compatibility test passed!")
    
    return True


def test_loss_integration():
    """Test that new loss function works in training loop."""
    print("\n" + "="*60)
    print("Testing Loss Integration...")
    print("="*60)
    
    from models.unet import UNet
    from training.losses import MSEWithPeakLoss
    
    # Create model and loss
    model = UNet(in_channels=3, num_keypoints=4, dropout_rate=0.1)
    criterion = MSEWithPeakLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Fake training step
    x = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 4, 128, 128)  # Fake heatmaps
    
    model.train()
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    
    print(f"Forward pass: OK")
    print(f"Loss computation: OK ({loss.item():.4f})")
    print(f"Backward pass: OK")
    print(f"Optimizer step: OK")
    
    print("✓ Loss integration test passed!")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Testing New Features from Old Project")
    print("="*60)
    
    tests = [
        ("MSE + Peak Loss", test_mse_peak_loss),
        ("Per-layer Dropout", test_per_layer_dropout),
        ("MC Dropout Uncertainty", test_mc_dropout_uncertainty),
        ("Backwards Compatibility", test_backwards_compatibility),
        ("Loss Integration", test_loss_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"✗ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results:
        icon = "✓" if status == "PASSED" else "✗"
        print(f"{icon} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
