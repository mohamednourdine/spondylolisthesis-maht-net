"""
Test and compare different loss functions for heatmap-based landmark detection.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from training.losses import (
    UNetKeypointLoss, 
    AdaptiveWingLoss, 
    MSEWithWeightedBackground,
    CombinedKeypointLoss
)


def create_test_heatmaps(batch_size=2, num_keypoints=4, height=512, width=512):
    """Create synthetic test heatmaps for evaluation."""
    
    # Create target heatmap with Gaussian blobs
    target = torch.zeros(batch_size, num_keypoints, height, width)
    
    # Add keypoints at known locations
    keypoint_locations = [
        (128, 128),  # Top-left region
        (384, 128),  # Top-right region
        (128, 384),  # Bottom-left region
        (384, 384),  # Bottom-right region
    ]
    
    sigma = 10
    for b in range(batch_size):
        for k, (cx, cy) in enumerate(keypoint_locations):
            # Add some variation per batch
            cx_offset = (b - batch_size // 2) * 20
            cy_offset = (b - batch_size // 2) * 20
            cx = cx + cx_offset
            cy = cy + cy_offset
            
            # Generate Gaussian
            y, x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing='ij'
            )
            gaussian = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            target[b, k] = gaussian
    
    return target, keypoint_locations


def test_loss_on_predictions(loss_fn, loss_name, target, error_levels):
    """Test loss function with different error levels."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {loss_name}")
    print(f"{'='*60}")
    
    losses = []
    
    for error_level in error_levels:
        # Create prediction with added noise
        if error_level == 0:
            # Perfect prediction
            pred = target.clone()
        else:
            # Add Gaussian noise
            noise = torch.randn_like(target) * error_level
            pred = torch.clamp(target + noise, 0, 1)
        
        # Convert to logits (inverse sigmoid)
        pred_logits = torch.logit(torch.clamp(pred, 1e-7, 1-1e-7))
        
        # Calculate loss
        loss = loss_fn(pred_logits, target)
        losses.append(loss.item())
        
        if error_level in [0, 0.1, 0.5]:
            print(f"  Error level {error_level:.2f}: Loss = {loss.item():.4f}")
    
    return losses


def compare_loss_functions():
    """Compare different loss functions."""
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON")
    print("="*80)
    
    # Create test data
    target, _ = create_test_heatmaps(batch_size=2, num_keypoints=4)
    
    # Different error levels
    error_levels = np.linspace(0, 1.0, 20)
    
    # Loss functions to test
    loss_functions = {
        'Focal Loss (CornerNet)': UNetKeypointLoss(use_focal=True),
        'Adaptive Wing Loss': AdaptiveWingLoss(),
        'Weighted MSE': MSEWithWeightedBackground(),
        'Combined (AWing+MSE)': CombinedKeypointLoss(),
        'Simple MSE': UNetKeypointLoss(use_focal=False)
    }
    
    # Test each loss function
    results = {}
    for name, loss_fn in loss_functions.items():
        losses = test_loss_on_predictions(loss_fn, name, target, error_levels)
        results[name] = losses
    
    # Plot comparison
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOT")
    print("="*80)
    
    plt.figure(figsize=(12, 8))
    
    for name, losses in results.items():
        plt.plot(error_levels, losses, marker='o', label=name, linewidth=2)
    
    plt.xlabel('Error Level (Noise Std)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Loss Function Comparison: Response to Prediction Errors', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = PROJECT_ROOT / 'tests' / 'loss_function_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Plot saved to: {output_path}")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. ADAPTIVE WING LOSS (RECOMMENDED)")
    print("   ✓ Best for precise landmark localization")
    print("   ✓ Focuses on small-medium errors (critical for medical imaging)")
    print("   ✓ Proven in facial landmark detection, works well for vertebrae")
    print("   ✓ Less sensitive to background/outliers")
    
    print("\n2. COMBINED LOSS (AWing + MSE)")
    print("   ✓ Balances precision and overall heatmap quality")
    print("   ✓ Good for challenging cases with occlusions")
    print("   ✓ Slightly slower but often more robust")
    
    print("\n3. FOCAL LOSS (Current)")
    print("   • Good for object detection")
    print("   • Handles class imbalance well")
    print("   - Less optimal for precise localization")
    print("   - May over-penalize medium errors")
    
    print("\n4. WEIGHTED MSE")
    print("   • Simple and fast")
    print("   • Good baseline")
    print("   - Not as good as Adaptive Wing for precision")
    
    print("\n5. SIMPLE MSE")
    print("   - Poor for heatmap regression")
    print("   - Background dominates the loss")
    print("   - Not recommended")


def test_loss_gradient_behavior():
    """Test gradient behavior of loss functions."""
    print("\n" + "="*80)
    print("GRADIENT BEHAVIOR TEST")
    print("="*80)
    
    # Create simple test case
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 32, 32] = 1.0  # Single keypoint at center
    
    # Create predictions with varying errors
    test_cases = [
        ('Perfect', target.clone()),
        ('Slight shift (2px)', torch.roll(target, shifts=(2, 2), dims=(2, 3))),
        ('Large shift (10px)', torch.roll(target, shifts=(10, 10), dims=(2, 3))),
        ('Lower confidence', target * 0.5),
    ]
    
    loss_fns = {
        'Focal': UNetKeypointLoss(use_focal=True),
        'AWing': AdaptiveWingLoss(),
        'Combined': CombinedKeypointLoss(),
    }
    
    print("\nGradient magnitudes for different scenarios:")
    print(f"{'Scenario':<20} {'Focal':>12} {'AWing':>12} {'Combined':>12}")
    print("-" * 60)
    
    for scenario, pred in test_cases:
        pred_logits = torch.logit(torch.clamp(pred, 1e-7, 1-1e-7))
        pred_logits.requires_grad = True
        
        grads = {}
        for name, loss_fn in loss_fns.items():
            if pred_logits.grad is not None:
                pred_logits.grad.zero_()
            
            loss = loss_fn(pred_logits, target)
            loss.backward()
            
            grad_magnitude = pred_logits.grad.abs().mean().item()
            grads[name] = grad_magnitude
        
        print(f"{scenario:<20} {grads['Focal']:>12.6f} {grads['AWing']:>12.6f} {grads['Combined']:>12.6f}")
    
    print("\nInterpretation:")
    print("  - Higher gradients = Stronger learning signal")
    print("  - Adaptive Wing should show strong gradients for slight shifts")
    print("  - This helps the model learn precise localization")


def main():
    """Run all loss function tests."""
    
    # Test 1: Compare loss functions
    compare_loss_functions()
    
    # Test 2: Gradient behavior
    test_loss_gradient_behavior()
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print("\nFor vertebral landmark detection, use:")
    print("  1st choice: AdaptiveWingLoss() - Best for precise localization")
    print("  2nd choice: CombinedKeypointLoss() - More robust, slightly slower")
    print("\nTo switch loss function, update config/mac_config.py:")
    print("  LOSS_FUNCTION = 'adaptive_wing'  # or 'combined' or 'focal'")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
