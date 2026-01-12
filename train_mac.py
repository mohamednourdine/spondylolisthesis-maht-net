#!/usr/bin/env python
"""
Mac-optimized training script.
Uses smaller images, minimal augmentation, and auto-detects MPS/CPU.
"""

import sys
import os
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.mac_config import MacConfig
from models.model_registry import ModelRegistry
from src.data.unet_dataset import UNetSpondylolisthesisDataset, unet_collate_fn
from src.data.augmentation import get_augmentation_pipeline
from training.losses import (
    UNetKeypointLoss, 
    AdaptiveWingLoss, 
    CombinedKeypointLoss,
    MSEWithWeightedBackground
)
from training.unet_trainer import UNetTrainer
from torch.utils.data import DataLoader


def get_device():
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Silicon GPU)")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("✓ Using CUDA GPU")
        return torch.device('cuda')
    else:
        print("⚠ Using CPU (slower)")
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Train on Mac (optimized)')
    parser.add_argument('--epochs', type=int, default=MacConfig.NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=MacConfig.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--image-size', type=int, default=MacConfig.IMAGE_SIZE[0],
                        help='Image size (default: 512)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable light augmentation (default: enabled)')
    args = parser.parse_args()
    
    print("="*60)
    print("Mac-Optimized Training")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Update config
    config = MacConfig
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = (args.image_size, args.image_size)
    config.USE_LIGHT_AUGMENTATION = not args.no_augmentation
    
    print(f"\nConfiguration:")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Light augmentation: {config.USE_LIGHT_AUGMENTATION}")
    print(f"  Base channels: {config.BASE_CHANNELS}")
    print("="*60)
    
    # Create experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"mac_{args.image_size}px"
    
    # Create augmentation for training
    train_augmentation = None
    if config.USE_LIGHT_AUGMENTATION:
        train_augmentation = get_augmentation_pipeline(
            mode='train', 
            aug_type='light', 
            image_size=config.IMAGE_SIZE
        )
        print(f"   Using light augmentation")
    
    # Create datasets with target image size
    print("\n1. Loading datasets...")
    
    train_dataset = UNetSpondylolisthesisDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        mode='train',
        heatmap_sigma=config.HEATMAP_SIGMA,
        heatmap_amplitude=config.HEATMAP_AMPLITUDE,
        output_stride=config.OUTPUT_STRIDE,
        target_size=config.IMAGE_SIZE,
        apply_clahe=config.APPLY_CLAHE,
        augmentation=train_augmentation,
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    val_dataset = UNetSpondylolisthesisDataset(
        image_dir=config.VAL_IMAGE_DIR,
        label_dir=config.VAL_LABEL_DIR,
        mode='val',
        heatmap_sigma=config.HEATMAP_SIGMA,
        heatmap_amplitude=config.HEATMAP_AMPLITUDE,
        output_stride=config.OUTPUT_STRIDE,
        target_size=config.IMAGE_SIZE,
        apply_clahe=False,
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    # Display detailed dataset information
    print(f"\n   {'='*56}")
    print(f"   DATASET SUMMARY")
    print(f"   {'='*56}")
    print(f"   Training Set:")
    print(f"     - Images: {len(train_dataset)} samples")
    print(f"     - Augmentation: {'✓ Enabled (light)' if train_augmentation else '✗ Disabled'}")
    if train_augmentation:
        print(f"       • Horizontal flips (50% probability)")
        print(f"       • Random rotation (±5°, 30% probability)")
        print(f"       • Each image seen {config.NUM_EPOCHS}x with variations")
        print(f"       • Effective training samples: ~{len(train_dataset) * config.NUM_EPOCHS} (with augmentation)")
    print(f"\n   Validation Set:")
    print(f"     - Images: {len(val_dataset)} samples")
    print(f"     - Augmentation: ✗ Disabled (consistent evaluation)")
    print(f"\n   Total unique images: {len(train_dataset) + len(val_dataset)}")
    print(f"   Train/Val split: {len(train_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}% / {len(val_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%")
    
    # Check for test dataset
    if config.TEST_IMAGE_DIR.exists() and config.TEST_LABEL_DIR.exists():
        test_images = list(config.TEST_IMAGE_DIR.glob('*.jpg')) + list(config.TEST_IMAGE_DIR.glob('*.png'))
        if test_images:
            print(f"\n   Test Set:")
            print(f"     - Images: {len(test_images)} samples")
            print(f"     - Location: {config.TEST_IMAGE_DIR.relative_to(config.PROJECT_ROOT)}")
            print(f"     - Note: Not used during training (for final evaluation)")
    
    print(f"   {'='*56}")

    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=unet_collate_fn,
        pin_memory=False  # Disable for Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=unet_collate_fn,
        pin_memory=False
    )
    
    # Create model (smaller version)
    print("\n2. Creating model...")
    model = ModelRegistry.create(
        'unet',
        in_channels=config.IN_CHANNELS,
        num_keypoints=config.NUM_KEYPOINTS,
        bilinear=config.BILINEAR,
        base_channels=config.BASE_CHANNELS
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params / 1e6:.2f}M")
    
    # Create loss and optimizer
    print("\n3. Setting up training...")
    
    # Select loss function based on config
    loss_func = config.LOSS_FUNCTION
    print(f"   Loss function: {loss_func}")
    
    if loss_func == 'adaptive_wing':
        criterion = AdaptiveWingLoss(
            omega=config.AWING_OMEGA,
            theta=config.AWING_THETA,
            epsilon=config.AWING_EPSILON,
            alpha=config.AWING_ALPHA
        )
    elif loss_func == 'combined':
        criterion = CombinedKeypointLoss()
    elif loss_func == 'weighted_mse':
        criterion = MSEWithWeightedBackground()
    elif loss_func == 'focal':
        criterion = UNetKeypointLoss(
            use_focal=True,
            focal_alpha=config.FOCAL_ALPHA,
            focal_beta=config.FOCAL_BETA
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
        verbose=True
    )
    
    # Create config dict for trainer
    trainer_config = {
        'model_name': 'unet',
        'training': {
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        },
        'image_size': config.IMAGE_SIZE,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
    }
    
    # Create trainer
    save_dir = config.RESULTS_DIR
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=trainer_config,
        save_dir=save_dir,
        scheduler=scheduler,
        experiment_name=exp_name
    )
    
    # Resume path
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        print(f"\n   Will resume from: {resume_path}")
    
    # Estimate training time (512x512 is 4x slower than 256x256)
    samples_per_epoch = len(train_dataset) + len(val_dataset)
    if device.type == 'mps':
        # ~0.08s per sample for 256px, ~0.32s for 512px
        time_per_sample = 0.08 * (args.image_size / 256) ** 2
    else:
        time_per_sample = 0.3 * (args.image_size / 256) ** 2
    estimated_time = (samples_per_epoch * config.NUM_EPOCHS * time_per_sample) / 60
    print(f"\n   Estimated training time: ~{estimated_time:.0f} minutes")
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.train(
        num_epochs=config.NUM_EPOCHS,
        resume_from=resume_path
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()
