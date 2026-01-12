#!/usr/bin/env python3
"""
Evaluate trained model on test set.
Generates predictions and visualizations for unlabeled test images.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import json
from scipy.ndimage import maximum_filter
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.mac_config import MacConfig as config
from models.unet import UNet
from src.data.preprocessing import ImagePreprocessor


# Colors for different corner types (TL, TR, BL, BR)
CORNER_COLORS = {
    0: '#FF0000',  # Top-Left: Red
    1: '#00FF00',  # Top-Right: Green
    2: '#0000FF',  # Bottom-Left: Blue
    3: '#FFFF00',  # Bottom-Right: Yellow
}
CORNER_NAMES = ['TL', 'TR', 'BL', 'BR']


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    model = UNet(
        in_channels=config.IN_CHANNELS,
        num_keypoints=config.NUM_KEYPOINTS,
        bilinear=config.BILINEAR,
        base_channels=config.BASE_CHANNELS,
        dropout_rate=0.0  # No dropout at inference
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess a single image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Resize
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Normalize to [-1, 1] (matching training)
    image_np = (image_np - 0.5) / 0.5
    
    # Convert to tensor [C, H, W]
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    return image_tensor, original_size


def extract_keypoints_from_heatmaps(heatmaps, max_vertebrae=10, threshold=0.3):
    """
    Extract keypoints from multi-channel heatmaps.
    
    Args:
        heatmaps: Predicted heatmaps [K, H, W] where K = num_vertebrae * 4
        max_vertebrae: Maximum number of vertebrae
        threshold: Confidence threshold
        
    Returns:
        keypoints: List of detected vertebrae with corners
    """
    detected_vertebrae = []
    
    for v in range(max_vertebrae):
        corners = []
        has_detection = False
        
        for c in range(4):  # 4 corners
            channel_idx = v * 4 + c
            if channel_idx >= heatmaps.shape[0]:
                corners.append({'x': 0, 'y': 0, 'confidence': 0})
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
                    corners.append({
                        'x': float(x_coords[0]),
                        'y': float(y_coords[0]),
                        'confidence': float(max_val)
                    })
                    has_detection = True
                else:
                    corners.append({'x': 0, 'y': 0, 'confidence': 0})
            else:
                corners.append({'x': 0, 'y': 0, 'confidence': 0})
        
        if has_detection:
            detected_vertebrae.append({
                'vertebra_idx': v,
                'corners': corners
            })
    
    return detected_vertebrae


def visualize_test_prediction(
    image_path: str,
    predictions: list,
    save_path: str = None
):
    """Visualize predictions on a test image."""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.set_title(f"Predictions: {Path(image_path).name}", fontsize=14)
    
    # Draw predictions
    for vert in predictions:
        corners = vert['corners']
        
        # Draw corners
        for c, corner in enumerate(corners):
            if corner['confidence'] > 0.3:
                ax.scatter(corner['x'], corner['y'], 
                          c=CORNER_COLORS[c], s=100, marker='x', 
                          linewidths=2, label=CORNER_NAMES[c] if vert['vertebra_idx'] == 0 else '')
        
        # Draw vertebra box if all corners detected
        visible_corners = [c for c in corners if c['confidence'] > 0.3]
        if len(visible_corners) >= 4:
            # Connect: TL -> TR -> BR -> BL -> TL
            order = [0, 1, 3, 2, 0]
            for i in range(4):
                if corners[order[i]]['confidence'] > 0.3 and corners[order[i+1]]['confidence'] > 0.3:
                    ax.plot(
                        [corners[order[i]]['x'], corners[order[i+1]]['x']],
                        [corners[order[i]]['y'], corners[order[i+1]]['y']],
                        'w-', linewidth=1.5, alpha=0.7
                    )
    
    # Add legend for first vertebra
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_test_evaluation(model, test_dir, device, save_dir):
    """Run evaluation on all test images."""
    
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    test_images = sorted(test_images)
    
    print(f"\nFound {len(test_images)} test images")
    
    # Create output directories
    save_dir = Path(save_dir)
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    all_predictions = {}
    
    for i, image_path in enumerate(test_images):
        print(f"  [{i+1}/{len(test_images)}] Processing {image_path.name}...")
        
        # Preprocess image
        image_tensor, original_size = preprocess_image(str(image_path), config.IMAGE_SIZE)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            pred_heatmaps = model(image_tensor)
        
        # Extract keypoints
        predictions = extract_keypoints_from_heatmaps(
            pred_heatmaps[0].cpu().numpy(),
            max_vertebrae=config.MAX_VERTEBRAE,
            threshold=0.3
        )
        
        # Store predictions
        all_predictions[image_path.name] = {
            'original_size': list(original_size),
            'processed_size': list(config.IMAGE_SIZE),
            'num_detected_vertebrae': len(predictions),
            'vertebrae': predictions
        }
        
        # Visualize
        vis_path = vis_dir / f"{image_path.stem}_prediction.png"
        visualize_test_prediction(str(image_path), predictions, save_path=str(vis_path))
        
        print(f"      Detected {len(predictions)} vertebrae")
    
    # Save predictions to JSON
    predictions_file = save_dir / 'test_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\n✓ Predictions saved to: {predictions_file}")
    
    # Generate summary
    print("\n" + "="*60)
    print("TEST SET EVALUATION SUMMARY")
    print("="*60)
    print(f"Total images:           {len(test_images)}")
    
    total_vertebrae = sum(p['num_detected_vertebrae'] for p in all_predictions.values())
    avg_vertebrae = total_vertebrae / len(test_images) if test_images else 0
    
    print(f"Total vertebrae found:  {total_vertebrae}")
    print(f"Avg vertebrae/image:    {avg_vertebrae:.1f}")
    print(f"Visualizations saved:   {vis_dir}")
    print("="*60)
    
    # Create summary CSV
    csv_file = save_dir / 'test_summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Num_Vertebrae', 'Original_Width', 'Original_Height'])
        for name, pred in all_predictions.items():
            writer.writerow([
                name,
                pred['num_detected_vertebrae'],
                pred['original_size'][0],
                pred['original_size'][1]
            ])
    
    print(f"Summary CSV: {csv_file}")
    
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=str, 
                       default='experiments/results/unet/mac_512px_20260112_210337/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, default=None,
                       help='Directory containing test images')
    parser.add_argument('--save-dir', type=str, default='experiments/test_evaluation',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Device
    device = torch.device(config.get_device())
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.PROJECT_ROOT / checkpoint_path
    
    model = load_model(checkpoint_path, device)
    
    # Test directory
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        test_dir = config.TEST_ROOT
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Save directory
    save_dir = config.PROJECT_ROOT / args.save_dir
    
    # Run evaluation
    predictions = run_test_evaluation(model, test_dir, device, save_dir)
    
    print(f"\n✓ Test evaluation complete!")
    print(f"  Results: {save_dir}")


if __name__ == '__main__':
    main()
