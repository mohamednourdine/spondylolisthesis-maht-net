#!/usr/bin/env python
"""
Quick test of the per-vertebra architecture.
Verifies shapes and basic functionality before training.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.mac_config import MacConfig
from models.model_registry import ModelRegistry
from src.data.unet_dataset import UNetSpondylolisthesisDataset, unet_collate_fn
from torch.utils.data import DataLoader
from evaluation.keypoint_evaluator import KeypointEvaluator


def test_architecture():
    """Test the per-vertebra architecture components."""
    
    print("="*60)
    print("Testing Per-Vertebra Architecture")
    print("="*60)
    
    config = MacConfig
    
    print(f"\n1. Configuration:")
    print(f"   MAX_VERTEBRAE: {config.MAX_VERTEBRAE}")
    print(f"   CORNERS_PER_VERTEBRA: {config.CORNERS_PER_VERTEBRA}")
    print(f"   NUM_KEYPOINTS (output channels): {config.NUM_KEYPOINTS}")
    print(f"   Expected: {config.MAX_VERTEBRAE} × {config.CORNERS_PER_VERTEBRA} = {config.MAX_VERTEBRAE * config.CORNERS_PER_VERTEBRA}")
    
    # Test dataset
    print(f"\n2. Testing Dataset...")
    dataset = UNetSpondylolisthesisDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        mode='train',
        heatmap_sigma=config.HEATMAP_SIGMA,
        output_stride=config.OUTPUT_STRIDE,
        target_size=config.IMAGE_SIZE,
        apply_clahe=False,
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\n3. Sample Data Shapes:")
    print(f"   Image: {sample['image'].shape}")
    print(f"   Heatmaps: {sample['heatmaps'].shape}")
    print(f"   Keypoints: {sample['keypoints'].shape}")
    print(f"   Expected heatmaps: [40, H, W] for 10 vertebrae × 4 corners")
    
    # Verify heatmap channels match expectation
    expected_channels = config.MAX_VERTEBRAE * config.CORNERS_PER_VERTEBRA
    actual_channels = sample['heatmaps'].shape[0]
    
    if actual_channels == expected_channels:
        print(f"   ✓ Heatmap channels correct: {actual_channels}")
    else:
        print(f"   ✗ ERROR: Expected {expected_channels} channels, got {actual_channels}")
        return False
    
    # Test DataLoader with collate_fn
    print(f"\n4. Testing DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=unet_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"   Batch images: {batch['images'].shape}")
    print(f"   Batch heatmaps: {batch['heatmaps'].shape}")
    print(f"   Batch keypoints: {len(batch['keypoints'])} samples")
    
    # Test model
    print(f"\n5. Testing Model...")
    model = ModelRegistry.create(
        'unet',
        in_channels=config.IN_CHANNELS,
        num_keypoints=config.NUM_KEYPOINTS,
        bilinear=config.BILINEAR,
        base_channels=config.BASE_CHANNELS
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params / 1e6:.2f}M")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_heatmaps = model(batch['images'])
    
    print(f"   Input shape: {batch['images'].shape}")
    print(f"   Output shape: {pred_heatmaps.shape}")
    print(f"   Expected: [B, 40, H, W]")
    
    if pred_heatmaps.shape[1] == expected_channels:
        print(f"   ✓ Model output channels correct: {pred_heatmaps.shape[1]}")
    else:
        print(f"   ✗ ERROR: Expected {expected_channels} output channels, got {pred_heatmaps.shape[1]}")
        return False
    
    # Test evaluator
    print(f"\n6. Testing Evaluator...")
    evaluator = KeypointEvaluator()
    
    pred_keypoints = evaluator.extract_keypoints_from_heatmaps(
        pred_heatmaps,
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    print(f"   Predicted keypoints (batch): {len(pred_keypoints)} samples")
    for i, kp in enumerate(pred_keypoints):
        print(f"   Sample {i}: {kp.shape} vertebrae detected")
    
    # Test metrics computation
    metrics = evaluator.evaluate_batch(
        pred_heatmaps,
        batch['heatmaps'],
        batch['keypoints'],
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    print(f"\n7. Metrics Computation:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n{'='*60}")
    print("✓ All tests passed! Architecture ready for training.")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = test_architecture()
    sys.exit(0 if success else 1)
