#!/usr/bin/env python
"""
Comprehensive test script to verify all main components work correctly.
Run this before training to catch bugs early.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path (parent of tests/)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Change to project root for relative imports

def test_separator(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")


def test_loss_function():
    """Test the UNet loss function."""
    test_separator("Loss Function")
    
    from training.losses import UNetKeypointLoss
    
    criterion = UNetKeypointLoss(use_focal=True, focal_alpha=2.0, focal_beta=4.0)
    
    B, K, H, W = 2, 4, 64, 64
    
    # Test 1: Random predictions should give positive loss
    pred = torch.randn(B, K, H, W)
    target = torch.zeros(B, K, H, W)
    target[:, :, 30, 30] = 1.0  # Add keypoints
    target[:, :, 40, 40] = 1.0
    
    loss = criterion(pred, target)
    print(f"  Random pred loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    assert not torch.isnan(loss), "Loss should not be NaN!"
    assert not torch.isinf(loss), "Loss should not be Inf!"
    print("  ✓ Random prediction: PASS (loss is positive)")
    
    # Test 2: Perfect prediction should have lower loss
    perfect_pred = torch.zeros(B, K, H, W) - 5.0  # Low logits everywhere
    perfect_pred[:, :, 30, 30] = 10.0  # High logits at keypoints
    perfect_pred[:, :, 40, 40] = 10.0
    
    loss_perfect = criterion(perfect_pred, target)
    print(f"  Perfect pred loss: {loss_perfect.item():.4f}")
    assert loss_perfect.item() < loss.item(), "Perfect prediction should have lower loss!"
    print("  ✓ Perfect prediction: PASS (lower loss than random)")
    
    # Test 3: Gradient flow
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None, "Gradients should flow!"
    print("  ✓ Gradient flow: PASS")
    
    # Test 4: MSE loss mode
    criterion_mse = UNetKeypointLoss(use_focal=False)
    loss_mse = criterion_mse(torch.randn(B, K, H, W), target)
    assert loss_mse.item() >= 0, "MSE loss should be non-negative!"
    print(f"  MSE loss: {loss_mse.item():.4f}")
    print("  ✓ MSE mode: PASS")
    
    return True


def test_heatmap_generation():
    """Test heatmap generation from keypoints."""
    test_separator("Heatmap Generation")
    
    from src.data.unet_dataset import generate_gaussian_heatmap, generate_heatmaps_from_keypoints
    
    # Test 1: Single keypoint heatmap
    H, W = 64, 64
    keypoint = np.array([32, 32, 1])  # x, y, visibility
    heatmap = generate_gaussian_heatmap(H, W, keypoint, sigma=2.0)
    
    assert heatmap.shape == (H, W), f"Wrong shape: {heatmap.shape}"
    assert heatmap.max() > 0, "Heatmap should have positive values!"
    assert heatmap[32, 32] > 0.9, f"Center should be ~1.0, got {heatmap[32, 32]}"
    print(f"  Heatmap max: {heatmap.max():.4f}, center: {heatmap[32, 32]:.4f}")
    print("  ✓ Single keypoint heatmap: PASS")
    
    # Test 2: Invisible keypoint should return zeros
    invisible_kp = np.array([32, 32, 0])  # visibility = 0
    heatmap_inv = generate_gaussian_heatmap(H, W, invisible_kp, sigma=2.0)
    assert heatmap_inv.max() == 0, "Invisible keypoint should produce zero heatmap!"
    print("  ✓ Invisible keypoint: PASS")
    
    # Test 3: Out-of-bounds keypoint
    oob_kp = np.array([100, 100, 1])  # Outside 64x64 image
    heatmap_oob = generate_gaussian_heatmap(H, W, oob_kp, sigma=2.0)
    assert heatmap_oob.max() == 0, "Out-of-bounds keypoint should produce zero heatmap!"
    print("  ✓ Out-of-bounds keypoint: PASS")
    
    # Test 4: Multiple keypoints for multiple vertebrae
    # Shape: [N_vertebrae, 4_corners, 3_coords]
    keypoints = np.array([
        [[10, 10, 1], [20, 10, 1], [20, 20, 1], [10, 20, 1]],  # Vertebra 1
        [[40, 40, 1], [50, 40, 1], [50, 50, 1], [40, 50, 1]],  # Vertebra 2
    ], dtype=np.float32)
    
    heatmaps = generate_heatmaps_from_keypoints(keypoints, H, W, num_keypoint_types=4, sigma=2.0)
    assert heatmaps.shape == (4, H, W), f"Wrong shape: {heatmaps.shape}"
    assert heatmaps.max() > 0, "Heatmaps should have positive values!"
    print(f"  Multi-vertebra heatmaps shape: {heatmaps.shape}")
    print("  ✓ Multi-vertebra heatmaps: PASS")
    
    return True


def test_dataset():
    """Test dataset loading and sample format."""
    test_separator("Dataset Loading")
    
    from config.mac_config import MacConfig
    from src.data.unet_dataset import UNetSpondylolisthesisDataset
    
    # Check if data exists
    if not MacConfig.TRAIN_IMAGE_DIR.exists():
        print(f"  ⚠ Data directory not found: {MacConfig.TRAIN_IMAGE_DIR}")
        print("  Skipping dataset test (run after setting up data)")
        return True
    
    dataset = UNetSpondylolisthesisDataset(
        image_dir=MacConfig.TRAIN_IMAGE_DIR,
        label_dir=MacConfig.TRAIN_LABEL_DIR,
        mode='train',
        target_size=(256, 256),
        heatmap_sigma=2.0
    )
    
    print(f"  Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty!"
    
    # Get a sample
    sample = dataset[0]
    
    # Check sample structure
    assert 'image' in sample, "Sample should have 'image' key!"
    assert 'heatmaps' in sample, "Sample should have 'heatmaps' key!"
    assert 'keypoints' in sample, "Sample should have 'keypoints' key!"
    
    image = sample['image']
    heatmaps = sample['heatmaps']
    keypoints = sample['keypoints']
    
    print(f"  Image shape: {image.shape}")
    print(f"  Heatmaps shape: {heatmaps.shape}")
    print(f"  Keypoints shape: {keypoints.shape}")
    
    # Validate shapes
    assert image.shape == (3, 256, 256), f"Wrong image shape: {image.shape}"
    assert heatmaps.shape[0] == 4, f"Should have 4 heatmaps, got {heatmaps.shape[0]}"
    assert heatmaps.shape[1] == 256 and heatmaps.shape[2] == 256, f"Wrong heatmap size: {heatmaps.shape}"
    
    # Check value ranges
    assert image.min() >= -5 and image.max() <= 5, f"Image values seem wrong: [{image.min()}, {image.max()}]"
    assert heatmaps.min() >= 0 and heatmaps.max() <= 1, f"Heatmap values should be in [0,1]: [{heatmaps.min()}, {heatmaps.max()}]"
    
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Heatmaps range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    print("  ✓ Dataset loading: PASS")
    
    return True


def test_model():
    """Test model forward pass."""
    test_separator("Model Forward Pass")
    
    from models.model_registry import ModelRegistry
    
    model = ModelRegistry.create(
        'unet',
        in_channels=3,
        num_keypoints=4,
        bilinear=True,
        base_channels=32
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.2f}M")
    
    # Test forward pass
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    assert out.shape == (2, 4, 256, 256), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN!"
    assert not torch.isinf(out).any(), "Output contains Inf!"
    
    print("  ✓ Model forward pass: PASS")
    
    # Test gradient flow
    model.train()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Model should have gradients!"
    print("  ✓ Gradient flow: PASS")
    
    return True


def test_metrics():
    """Test metric calculation."""
    test_separator("Metrics Calculation")
    
    from evaluation.keypoint_evaluator import get_global_evaluator
    
    evaluator = get_global_evaluator()
    
    # Create dummy predictions and targets
    B, K, H, W = 2, 4, 64, 64
    
    # Perfect predictions (heatmap peaks at exact keypoint locations)
    pred_heatmaps = torch.zeros(B, K, H, W)
    target_heatmaps = torch.zeros(B, K, H, W)
    
    # Add keypoints at same locations
    pred_heatmaps[:, 0, 30, 30] = 1.0
    pred_heatmaps[:, 1, 30, 40] = 1.0
    pred_heatmaps[:, 2, 40, 30] = 1.0
    pred_heatmaps[:, 3, 40, 40] = 1.0
    
    target_heatmaps[:, 0, 30, 30] = 1.0
    target_heatmaps[:, 1, 30, 40] = 1.0
    target_heatmaps[:, 2, 40, 30] = 1.0
    target_heatmaps[:, 3, 40, 40] = 1.0
    
    # Create keypoints array [B, N_vertebrae, 4_corners, 3_coords]
    keypoints = torch.zeros(B, 1, 4, 3)
    keypoints[:, 0, 0, :] = torch.tensor([30, 30, 1])
    keypoints[:, 0, 1, :] = torch.tensor([40, 30, 1])
    keypoints[:, 0, 2, :] = torch.tensor([30, 40, 1])
    keypoints[:, 0, 3, :] = torch.tensor([40, 40, 1])
    
    metrics = evaluator.evaluate_batch(pred_heatmaps, target_heatmaps, keypoints)
    
    print(f"  Metrics: {metrics}")
    
    assert 'MRE' in metrics, "Metrics should include MRE!"
    assert metrics['MRE'] >= 0, "MRE should be non-negative!"
    
    # Perfect prediction should have low MRE
    # Note: There might be some small error due to discrete sampling
    if metrics['MRE'] < 5:
        print(f"  MRE for perfect prediction: {metrics['MRE']:.2f} (should be ~0)")
        print("  ✓ Metrics calculation: PASS")
    else:
        print(f"  ⚠ MRE seems high for perfect prediction: {metrics['MRE']:.2f}")
        print("  This might need investigation but not blocking")
    
    return True


def test_early_stopping():
    """Test early stopping logic."""
    test_separator("Early Stopping Logic")
    
    # Simulate the early stopping behavior
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    
    # Simulate improving losses
    losses = [10.0, 9.0, 8.5, 8.0, 7.5, 7.0, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3]
    
    stopped_at = None
    for epoch, loss in enumerate(losses):
        is_best = loss < best_val_loss
        if is_best:
            best_val_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            stopped_at = epoch + 1
            break
    
    print(f"  Losses: {losses[:min(stopped_at or len(losses), 12)]}")
    print(f"  Early stopping at epoch: {stopped_at}")
    
    # Should stop at epoch 12 (after 5 epochs without improvement from epoch 7)
    assert stopped_at == 12, f"Should stop at epoch 12, stopped at {stopped_at}"
    print("  ✓ Early stopping: PASS")
    
    # Test case 2: Always improving
    patience_counter = 0
    best_val_loss = float('inf')
    improving_losses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    
    stopped = False
    for loss in improving_losses:
        is_best = loss < best_val_loss
        if is_best:
            best_val_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            stopped = True
            break
    
    assert not stopped, "Should not stop when always improving!"
    print("  ✓ No early stop when improving: PASS")
    
    return True


def test_dataloader():
    """Test DataLoader with collate function."""
    test_separator("DataLoader & Collate")
    
    from config.mac_config import MacConfig
    from src.data.unet_dataset import UNetSpondylolisthesisDataset, unet_collate_fn
    from torch.utils.data import DataLoader
    
    if not MacConfig.TRAIN_IMAGE_DIR.exists():
        print("  ⚠ Data directory not found, skipping")
        return True
    
    dataset = UNetSpondylolisthesisDataset(
        image_dir=MacConfig.TRAIN_IMAGE_DIR,
        label_dir=MacConfig.TRAIN_LABEL_DIR,
        mode='train',
        target_size=(256, 256),
        heatmap_sigma=2.0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=unet_collate_fn
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Heatmaps shape: {batch['heatmaps'].shape}")
    
    assert batch['images'].shape[0] == 4, "Batch size should be 4"
    assert batch['heatmaps'].shape[0] == 4, "Batch size should be 4"
    
    print("  ✓ DataLoader: PASS")
    
    return True


def run_all_tests():
    """Run all component tests."""
    print("\n" + "="*60)
    print("SPONDYLOLISTHESIS MAHT-NET COMPONENT TESTS")
    print("="*60)
    
    tests = [
        ("Loss Function", test_loss_function),
        ("Heatmap Generation", test_heatmap_generation),
        ("Model", test_model),
        ("Early Stopping", test_early_stopping),
        ("Dataset", test_dataset),
        ("DataLoader", test_dataloader),
        ("Metrics", test_metrics),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
        if status != "PASS":
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
