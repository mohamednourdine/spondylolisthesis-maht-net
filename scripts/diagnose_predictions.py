"""
Diagnostic script to visualize model predictions and identify issues.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from models.unet import UNet
from src.data.unet_dataset import UNetSpondylolisthesisDataset
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import LightAugmentation
from config.mac_config import MacConfig
from evaluation.keypoint_evaluator import KeypointEvaluator


def load_model(checkpoint_path):
    """Load trained model from checkpoint."""
    model = UNet(
        in_channels=3,
        num_keypoints=MacConfig.NUM_KEYPOINTS,
        base_channels=MacConfig.BASE_CHANNELS,
        bilinear=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='mps')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('mps')
    model.eval()
    
    return model


def visualize_sample(model, dataset, idx, save_dir):
    """Visualize a single sample with predictions."""
    print(f"\n{'='*60}")
    print(f"Analyzing sample {idx}")
    print(f"{'='*60}")
    
    # Get data
    sample = dataset[idx]
    image = sample['image'].unsqueeze(0).to('mps')
    gt_heatmaps = sample['heatmaps'].numpy()  # [K, H, W]
    gt_keypoints_raw = sample['keypoints'].numpy()  # [N_vertebrae, 4, 3]
    
    # Flatten keypoints: take first vertebra's keypoints for visualization
    # gt_keypoints_raw has shape [N_vert, 4, 3] where 4 is corners, 3 is [x, y, vis]
    # For visualization, we'll look at all vertebrae but focus on keypoint type
    num_vertebrae = gt_keypoints_raw.shape[0]
    num_kp_per_vert = gt_keypoints_raw.shape[1]
    
    print(f"\nInput shape: {image.shape}")
    print(f"GT heatmaps shape: {gt_heatmaps.shape}")
    print(f"GT keypoints shape: {gt_keypoints_raw.shape}")
    print(f"Number of vertebrae: {num_vertebrae}")
    print(f"Keypoints per vertebra: {num_kp_per_vert}")
    
    # Get predictions
    with torch.no_grad():
        pred_heatmaps = model(image)
    
    pred_heatmaps_np = pred_heatmaps[0].cpu().numpy()  # [K, H, W]
    
    print(f"Pred heatmaps shape: {pred_heatmaps_np.shape}")
    print(f"\nPred heatmap stats:")
    print(f"  Min: {pred_heatmaps_np.min():.2f}")
    print(f"  Max: {pred_heatmaps_np.max():.2f}")
    print(f"  Mean: {pred_heatmaps_np.mean():.2f}")
    
    print(f"\nGT heatmap stats:")
    print(f"  Min: {gt_heatmaps.min():.2f}")
    print(f"  Max: {gt_heatmaps.max():.2f}")
    print(f"  Mean: {gt_heatmaps.mean():.2f}")
    
    # Extract predicted keypoints using argmax
    evaluator = KeypointEvaluator(
        sdr_thresholds_px=[2.0, 4.0, 8.0, 16.0]
    )
    pred_keypoints_list = evaluator.extract_keypoints_from_heatmaps(pred_heatmaps)
    pred_keypoints = pred_keypoints_list[0]  # [K, 2]
    
    print(f"\n{'='*60}")
    print("Keypoint Comparison (showing one vertebra's corners):")
    print(f"{'='*60}")
    print(f"{'Type':<15} {'GT_X':>8} {'GT_Y':>8} {'Pred_X':>8} {'Pred_Y':>8} {'Error':>8}")
    print(f"{'-'*60}")
    
    errors = []
    # Show first visible vertebra's keypoints
    for vert_idx in range(num_vertebrae):
        for kp_idx in range(num_kp_per_vert):
            gt_x, gt_y, vis = gt_keypoints_raw[vert_idx, kp_idx]
            
            # Predicted keypoints are per keypoint type (4 heatmaps)
            # Each heatmap should find that keypoint type across all vertebrae
            pred_x, pred_y = pred_keypoints[kp_idx]
            
            if vis > 0:
                error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                errors.append(error)
                kp_label = f"V{vert_idx}_KP{kp_idx}"
                print(f"{kp_label:<15} {gt_x:8.1f} {gt_y:8.1f} {pred_x:8.1f} {pred_y:8.1f} {error:8.1f}")
        
        # Only show first vertebra for clarity
        if vert_idx == 0:
            break
    
    if errors:
        print(f"{'-'*60}")
        print(f"Mean Error: {np.mean(errors):.2f} pixels")
        print(f"{'='*60}")
    
    # Create visualization
    num_keypoints = MacConfig.NUM_KEYPOINTS
    fig, axes = plt.subplots(3, num_keypoints, figsize=(3*num_keypoints, 9))
    
    # Denormalize image for display
    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * 0.5 + 0.5).clip(0, 1)  # [-1,1] -> [0,1]
    
    for k in range(num_keypoints):
        # Row 1: Ground truth heatmap - show all GT keypoints of this type
        axes[0, k].imshow(image_np, cmap='gray')
        axes[0, k].imshow(gt_heatmaps[k], alpha=0.6, cmap='jet')
        # Plot all vertebrae's keypoints of this type
        for vert_idx in range(num_vertebrae):
            if gt_keypoints_raw[vert_idx, k, 2] > 0:  # visible
                axes[0, k].plot(gt_keypoints_raw[vert_idx, k, 0], gt_keypoints_raw[vert_idx, k, 1], 
                              'g+', markersize=10, markeredgewidth=2)
        axes[0, k].set_title(f'GT Type{k}')
        axes[0, k].axis('off')
        
        # Row 2: Predicted heatmap
        axes[1, k].imshow(image_np, cmap='gray')
        axes[1, k].imshow(pred_heatmaps_np[k], alpha=0.6, cmap='jet')
        axes[1, k].plot(pred_keypoints[k, 0], pred_keypoints[k, 1], 'r+', markersize=15, markeredgewidth=3)
        axes[1, k].set_title(f'Pred Type{k}')
        axes[1, k].axis('off')
        
        # Row 3: Overlay comparison - compare with first visible vertebra
        axes[2, k].imshow(image_np, cmap='gray')
        for vert_idx in range(num_vertebrae):
            if gt_keypoints_raw[vert_idx, k, 2] > 0:
                axes[2, k].plot(gt_keypoints_raw[vert_idx, k, 0], gt_keypoints_raw[vert_idx, k, 1], 
                              'g+', markersize=10, markeredgewidth=2, label='GT' if vert_idx == 0 else '')
        axes[2, k].plot(pred_keypoints[k, 0], pred_keypoints[k, 1], 'r+', markersize=15, markeredgewidth=3, label='Pred')
        # Calculate error to closest GT keypoint of this type
        min_error = float('inf')
        for vert_idx in range(num_vertebrae):
            if gt_keypoints_raw[vert_idx, k, 2] > 0:
                error = np.sqrt((gt_keypoints_raw[vert_idx, k, 0] - pred_keypoints[k, 0])**2 + 
                              (gt_keypoints_raw[vert_idx, k, 1] - pred_keypoints[k, 1])**2)
                min_error = min(min_error, error)
        if min_error != float('inf'):
            axes[2, k].set_title(f'Min Error: {min_error:.1f}px')
        axes[2, k].legend(loc='upper right', fontsize=8)
        axes[2, k].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'diagnosis_sample_{idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    
    return np.mean(errors) if errors else 0


def main():
    # Find latest checkpoint
    results_dir = Path('/Users/mnourdine/phd/spondylolisthesis-maht-net/experiments/results/unet')
    latest_exp = sorted(results_dir.glob('calibrated_model_*'))[-1]
    checkpoint_path = latest_exp / 'best_model.pth'
    
    print(f"\n{'='*60}")
    print("Model Prediction Diagnosis")
    print(f"{'='*60}")
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path)
    print("✓ Model loaded")
    
    # Create dataset
    print("\nLoading validation dataset...")
    preprocessor = ImagePreprocessor(target_size=MacConfig.IMAGE_SIZE)
    augmentation = LightAugmentation(mode='val')
    
    dataset = UNetSpondylolisthesisDataset(
        image_dir=MacConfig.VAL_IMAGE_DIR,
        label_dir=MacConfig.VAL_LABEL_DIR,
        mode='val',
        preprocessor=preprocessor,
        augmentation=augmentation,
        heatmap_sigma=MacConfig.HEATMAP_SIGMA
    )
    print(f"✓ Loaded {len(dataset)} validation samples")
    
    # Create output directory
    save_dir = latest_exp / 'diagnosis'
    save_dir.mkdir(exist_ok=True)
    
    # Analyze multiple samples
    num_samples = min(5, len(dataset))
    print(f"\nAnalyzing {num_samples} samples...")
    
    all_errors = []
    for i in range(num_samples):
        error = visualize_sample(model, dataset, i, save_dir)
        all_errors.append(error)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Average error across {num_samples} samples: {np.mean(all_errors):.2f} pixels")
    print(f"Min error: {np.min(all_errors):.2f} pixels")
    print(f"Max error: {np.max(all_errors):.2f} pixels")
    print(f"\nDiagnosis images saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
