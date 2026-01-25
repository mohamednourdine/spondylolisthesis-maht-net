#!/usr/bin/env python
"""
Training script for ResNet-50 heatmap model.
Uses pretrained ImageNet weights with a simple deconv decoder.

Key differences from UNet training:
- Differential learning rates (lower for backbone)
- ImageNet normalization
- AdamW optimizer
- Layer freezing for pretrained backbone
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
from training.losses import MSEWithWeightedBackground
from training.unet_trainer import UNetTrainer  # Can reuse since output format is same
from evaluation.keypoint_evaluator import get_global_evaluator
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
    parser = argparse.ArgumentParser(description='Train ResNet-50 Heatmap Model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size (default: 512)')
    
    # Learning rates
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for decoder/head (default: 1e-4)')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Learning rate for backbone (default: 1e-5, 10x smaller)')
    
    # Model settings
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights (default: True)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do NOT use pretrained weights')
    parser.add_argument('--freeze-layers', type=int, default=2,
                        help='Number of backbone layers to freeze (0-4, default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for decoder (default: 0.3)')
    
    # Other
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable augmentation')
    
    args = parser.parse_args()
    
    # Handle pretrained flag
    use_pretrained = args.pretrained and not args.no_pretrained
    
    print("="*60)
    print("ResNet-50 Heatmap Training")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Configuration
    config = MacConfig
    
    print(f"\nConfiguration:")
    print(f"  Image size: {args.image_size}×{args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Pretrained: {use_pretrained}")
    print(f"  Freeze layers: {args.freeze_layers}")
    print(f"  Decoder LR: {args.lr}")
    print(f"  Backbone LR: {args.backbone_lr}")
    print(f"  Dropout: {args.dropout}")
    print("="*60)
    
    # Create experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        pretrained_str = "pretrained" if use_pretrained else "scratch"
        exp_name = f"resnet50_{pretrained_str}"
    
    # Create augmentation for training
    use_augmentation = not args.no_augmentation
    train_augmentation = None
    if use_augmentation:
        train_augmentation = get_augmentation_pipeline(
            mode='train', 
            aug_type='light', 
            image_size=(args.image_size, args.image_size)
        )
        print(f"✓ Using light augmentation")
    
    # Create datasets
    print("\n1. Loading datasets...")
    
    train_dataset = UNetSpondylolisthesisDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        mode='train',
        heatmap_sigma=config.HEATMAP_SIGMA,
        heatmap_amplitude=config.HEATMAP_AMPLITUDE,
        output_stride=config.OUTPUT_STRIDE,
        target_size=(args.image_size, args.image_size),
        apply_clahe=False,
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
        target_size=(args.image_size, args.image_size),
        apply_clahe=False,
        max_vertebrae=config.MAX_VERTEBRAE,
        corners_per_vertebra=config.CORNERS_PER_VERTEBRA
    )
    
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=unet_collate_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=unet_collate_fn,
        pin_memory=False
    )
    
    # Create model
    print("\n2. Creating ResNet-50 model...")
    model = ModelRegistry.create(
        'resnet',
        in_channels=3,
        num_keypoints=config.NUM_KEYPOINTS,
        pretrained=use_pretrained,
        freeze_backbone_layers=args.freeze_layers,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Create loss function (same as UNet for fair comparison)
    print("\n3. Setting up training...")
    criterion = MSEWithWeightedBackground()
    print(f"   Loss: Weighted MSE (background=0.05, keypoint=5.0)")
    
    # Create optimizer with differential learning rates
    param_groups = model.get_parameter_groups(
        backbone_lr=args.backbone_lr,
        decoder_lr=args.lr
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4
    )
    print(f"   Optimizer: AdamW with differential LR")
    print(f"     - Backbone: lr={args.backbone_lr}")
    print(f"     - Decoder/Head: lr={args.lr}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=True
    )
    
    # Initialize evaluator
    evaluator = get_global_evaluator(
        sdr_thresholds_px=config.SDR_THRESHOLDS_PX
    )
    print(f"   SDR thresholds (px): {config.SDR_THRESHOLDS_PX}")
    
    # Create config dict for trainer
    trainer_config = {
        'model_name': 'resnet',
        'training': {
            'early_stopping_patience': 15,  # More patience for pretrained
        },
        'image_size': (args.image_size, args.image_size),
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'backbone_lr': args.backbone_lr,
        'pretrained': use_pretrained,
        'freeze_layers': args.freeze_layers,
    }
    
    # Create trainer (reuse UNetTrainer since output format is identical)
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
    
    # Resume if specified
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        print(f"\n   Will resume from: {resume_path}")
    
    # Estimate training time
    samples_per_epoch = len(train_dataset) + len(val_dataset)
    # ResNet is ~1.5x slower than UNet
    time_per_sample = 0.12 * (args.image_size / 256) ** 2
    estimated_time = (samples_per_epoch * args.epochs * time_per_sample) / 60
    print(f"\n   Estimated training time: ~{estimated_time:.0f} minutes ({estimated_time/60:.1f} hours)")
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.train(
        num_epochs=args.epochs,
        resume_from=resume_path
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {trainer.save_dir}")
    print(f"\nTo evaluate on test set:")
    print(f"  python scripts/evaluate_test.py --model-path {trainer.save_dir / 'best_model.pth'} --model-type resnet")


if __name__ == '__main__':
    main()
