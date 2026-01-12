#!/usr/bin/env python3
"""
Visualize model predictions on validation images.
Shows predicted vs ground truth keypoints overlaid on images.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.mac_config import MacConfig as config
from models.unet import UNet
from src.data.unet_dataset import UNetSpondylolisthesisDataset
from src.data.augmentation import LightAugmentation
from scipy.ndimage import maximum_filter


# Colors for different corner types (TL, TR, BL, BR)
CORNER_COLORS = {
    0: '#FF0000',  # Top-Left: Red
    1: '#00FF00',  # Top-Right: Green
    2: '#0000FF',  # Bottom-Left: Blue
    3: '#FFFF00',  # Bottom-Right: Yellow
}

CORNER_NAMES = ['TL', 'TR', 'BL', 'BR']


def extract_keypoints_from_heatmaps(heatmaps, num_vertebrae, threshold=0.3):
    """
    Extract keypoints from multi-channel heatmaps.
    
    Args:
        heatmaps: Predicted heatmaps [K, H, W] where K = num_vertebrae * 4
        num_vertebrae: Number of vertebrae to extract
        threshold: Confidence threshold
        
    Returns:
        keypoints: Array [num_vertebrae, 4, 3] (x, y, confidence)
    """
    keypoints = np.zeros((num_vertebrae, 4, 3))
    
    for v in range(num_vertebrae):
        for c in range(4):  # 4 corners
            channel_idx = v * 4 + c
            if channel_idx >= heatmaps.shape[0]:
                continue
                
            heatmap = heatmaps[channel_idx]
            
            # Find peak location
            max_val = heatmap.max()
            if max_val > threshold:
                # Apply NMS
                local_max = maximum_filter(heatmap, size=3) == heatmap
                peaks = (heatmap == max_val) & local_max
                y_coords, x_coords = np.where(peaks)
                
                if len(x_coords) > 0:
                    # Take the first peak (highest confidence)
                    keypoints[v, c, 0] = x_coords[0]
                    keypoints[v, c, 1] = y_coords[0]
                    keypoints[v, c, 2] = max_val
    
    return keypoints


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model with same config as training
    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_keypoints=config.NUM_KEYPOINTS,
        bilinear=config.BILINEAR,
        base_channels=config.BASE_CHANNELS,
        dropout_rate=0.0  # No dropout at inference
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    # Tensor is normalized with mean=0.5, std=0.5
    # So: normalized = (img - 0.5) / 0.5 => img = normalized * 0.5 + 0.5
    img = tensor.cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = img * 0.5 + 0.5  # Denormalize
    img = np.clip(img, 0, 1)
    return img


def visualize_single_image(
    image: np.ndarray,
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    save_path: str = None,
    title: str = "Predictions vs Ground Truth"
):
    """
    Visualize predictions for a single image.
    
    Args:
        image: Image array [H, W, C]
        pred_keypoints: Predicted keypoints [N_vertebrae, 4, 3] (x, y, conf)
        gt_keypoints: Ground truth keypoints [N_vertebrae, 4, 3]
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Ground Truth only
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth", fontsize=14)
    for v in range(gt_keypoints.shape[0]):
        for c in range(4):
            if gt_keypoints[v, c, 2] > 0.5:  # Visible
                x, y = gt_keypoints[v, c, :2]
                axes[0].scatter(x, y, c=CORNER_COLORS[c], s=60, marker='o', 
                              edgecolors='white', linewidths=1)
                if v == 0:  # Legend only for first vertebra
                    axes[0].scatter([], [], c=CORNER_COLORS[c], label=CORNER_NAMES[c])
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].axis('off')
    
    # 2. Predictions only
    axes[1].imshow(image)
    axes[1].set_title("Predictions", fontsize=14)
    for v in range(pred_keypoints.shape[0]):
        for c in range(4):
            if pred_keypoints[v, c, 2] > 0.3:  # Confidence threshold
                x, y = pred_keypoints[v, c, :2]
                axes[1].scatter(x, y, c=CORNER_COLORS[c], s=60, marker='x', 
                              linewidths=2)
    axes[1].axis('off')
    
    # 3. Overlay with connections
    axes[2].imshow(image)
    axes[2].set_title("Overlay (○=GT, ×=Pred)", fontsize=14)
    
    for v in range(min(gt_keypoints.shape[0], pred_keypoints.shape[0])):
        for c in range(4):
            # Ground truth
            if gt_keypoints[v, c, 2] > 0.5:
                gt_x, gt_y = gt_keypoints[v, c, :2]
                axes[2].scatter(gt_x, gt_y, c=CORNER_COLORS[c], s=80, marker='o',
                              edgecolors='white', linewidths=1.5, alpha=0.8)
                
                # Prediction
                if pred_keypoints[v, c, 2] > 0.3:
                    pred_x, pred_y = pred_keypoints[v, c, :2]
                    axes[2].scatter(pred_x, pred_y, c=CORNER_COLORS[c], s=80, marker='x',
                                  linewidths=2, alpha=0.8)
                    
                    # Draw line connecting GT to Pred
                    axes[2].plot([gt_x, pred_x], [gt_y, pred_y], 
                               c=CORNER_COLORS[c], linewidth=1, alpha=0.5)
    
    # Draw vertebra boxes for GT
    for v in range(gt_keypoints.shape[0]):
        if np.all(gt_keypoints[v, :, 2] > 0.5):  # All corners visible
            corners = gt_keypoints[v, :, :2]
            # Connect corners: TL->TR->BR->BL->TL
            order = [0, 1, 3, 2, 0]
            for i in range(4):
                axes[2].plot(
                    [corners[order[i], 0], corners[order[i+1], 0]],
                    [corners[order[i], 1], corners[order[i+1], 1]],
                    'w-', linewidth=1, alpha=0.5
                )
    
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def run_inference(model, dataset, device, num_samples=10, save_dir=None):
    """Run inference on dataset samples and visualize."""
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"\nVisualizing {len(indices)} samples...")
    
    errors = []
    
    for i, idx in enumerate(indices):
        # Get sample
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_keypoints = sample['keypoints'].numpy()  # [N_vertebrae, 4, 3]
        
        # Run inference
        with torch.no_grad():
            pred_heatmaps = model(image)
        
        # Extract keypoints from heatmaps
        pred_keypoints = extract_keypoints_from_heatmaps(
            pred_heatmaps[0].cpu().numpy(),
            num_vertebrae=gt_keypoints.shape[0],
            threshold=0.3
        )
        
        # Denormalize image for display
        display_image = denormalize_image(sample['image'])
        
        # Calculate error for this sample
        visible_mask = gt_keypoints[:, :, 2] > 0.5
        if visible_mask.sum() > 0:
            gt_visible = gt_keypoints[visible_mask][:, :2]
            pred_visible = pred_keypoints[visible_mask][:, :2]
            sample_error = np.sqrt(((gt_visible - pred_visible) ** 2).sum(axis=1)).mean()
            errors.append(sample_error)
        else:
            sample_error = float('nan')
        
        # Create title with error
        title = f"Sample {idx} | MRE: {sample_error:.1f} px"
        
        # Save path
        save_path = save_dir / f"sample_{idx:04d}.png" if save_dir else None
        
        # Visualize
        visualize_single_image(
            display_image,
            pred_keypoints,
            gt_keypoints,
            save_path=save_path,
            title=title
        )
        
        print(f"  [{i+1}/{len(indices)}] Sample {idx}: MRE = {sample_error:.1f} px")
    
    # Summary
    if errors:
        print(f"\n" + "="*50)
        print(f"Summary ({len(errors)} samples):")
        print(f"  Mean MRE:   {np.mean(errors):.1f} px")
        print(f"  Median MRE: {np.median(errors):.1f} px")
        print(f"  Best:       {np.min(errors):.1f} px")
        print(f"  Worst:      {np.max(errors):.1f} px")
        print(f"="*50)


def visualize_heatmaps(model, dataset, device, sample_idx=0, save_path=None):
    """Visualize predicted heatmaps for a single sample."""
    
    sample = dataset[sample_idx]
    image = sample['image'].unsqueeze(0).to(device)
    target_heatmaps = sample['heatmaps'].numpy()  # [K, H, W]
    
    with torch.no_grad():
        pred_heatmaps = model(image)[0].cpu().numpy()  # [K, H, W]
    
    # Display first 8 channels (2 vertebrae x 4 corners)
    num_channels = min(8, pred_heatmaps.shape[0])
    
    fig, axes = plt.subplots(3, num_channels, figsize=(20, 8))
    
    # Row 0: Image with GT keypoints
    display_image = denormalize_image(sample['image'])
    
    for c in range(num_channels):
        # Row 0: Target heatmaps
        axes[0, c].imshow(target_heatmaps[c], cmap='hot')
        axes[0, c].set_title(f"Target Ch{c}", fontsize=10)
        axes[0, c].axis('off')
        
        # Row 1: Predicted heatmaps
        axes[1, c].imshow(pred_heatmaps[c], cmap='hot')
        axes[1, c].set_title(f"Pred Ch{c}", fontsize=10)
        axes[1, c].axis('off')
        
        # Row 2: Difference
        diff = np.abs(pred_heatmaps[c] - target_heatmaps[c])
        axes[2, c].imshow(diff, cmap='coolwarm')
        axes[2, c].set_title(f"Diff Ch{c}", fontsize=10)
        axes[2, c].axis('off')
    
    plt.suptitle(f"Heatmap Comparison (Sample {sample_idx})", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap visualization: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, 
                       default='experiments/results/unet/mac_512px_20260112_210337/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default='experiments/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--heatmaps', action='store_true',
                       help='Also visualize heatmaps')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='Dataset split to use')
    args = parser.parse_args()
    
    # Device
    device = torch.device(config.get_device())
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.PROJECT_ROOT / checkpoint_path
    
    model = load_model(checkpoint_path, device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    augmentation = LightAugmentation(mode='val')  # No augmentation for visualization
    
    if args.split == 'val':
        image_dir = config.VAL_IMAGE_DIR
        label_dir = config.VAL_LABEL_DIR
    else:
        image_dir = config.TRAIN_IMAGE_DIR
        label_dir = config.TRAIN_LABEL_DIR
    
    dataset = UNetSpondylolisthesisDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        target_size=config.IMAGE_SIZE,
        augmentation=augmentation,
        max_vertebrae=config.MAX_VERTEBRAE,
        heatmap_sigma=config.HEATMAP_SIGMA,
        heatmap_amplitude=config.HEATMAP_AMPLITUDE,
        mode=args.split
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create save directory
    save_dir = config.PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    run_inference(model, dataset, device, num_samples=args.num_samples, save_dir=save_dir)
    
    # Optionally visualize heatmaps
    if args.heatmaps:
        print("\nGenerating heatmap visualizations...")
        heatmap_dir = save_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        
        for i in range(min(3, len(dataset))):
            visualize_heatmaps(
                model, dataset, device, 
                sample_idx=i,
                save_path=heatmap_dir / f"heatmaps_sample_{i}.png"
            )
    
    print(f"\n✓ Visualizations saved to: {save_dir}")


if __name__ == '__main__':
    main()
