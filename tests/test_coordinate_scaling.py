"""
Test coordinate scaling when resizing images.
CRITICAL: When we resize images from original size to 512x512,
keypoint coordinates MUST be scaled proportionally.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data.preprocessing import ImagePreprocessor, KeypointTransformer
from src.data.unet_dataset import UNetSpondylolisthesisDataset, generate_gaussian_heatmap
from config.mac_config import MacConfig


def test_coordinate_scaling():
    """Test that coordinates are properly scaled when resizing images."""
    print("\n" + "="*60)
    print("TEST 1: Coordinate Scaling with Image Resize")
    print("="*60)
    
    # Create a test image and keypoint
    original_size = (1024, 768)  # width, height
    target_size = (512, 512)
    
    # Create dummy image
    test_image = np.random.randint(0, 255, (original_size[1], original_size[0], 3), dtype=np.uint8)
    
    # Test keypoint at center of original image
    original_keypoint_x = original_size[0] / 2  # 512
    original_keypoint_y = original_size[1] / 2  # 384
    
    print(f"Original image size: {original_size} (W×H)")
    print(f"Target image size: {target_size} (W×H)")
    print(f"Original keypoint: ({original_keypoint_x:.1f}, {original_keypoint_y:.1f})")
    
    # Preprocess image
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=False)
    processed_image, transform_info = preprocessor.resize_with_padding(test_image)
    
    print(f"\nTransform info:")
    print(f"  Scale: {transform_info['scale']:.4f}")
    print(f"  X offset: {transform_info['x_offset']}")
    print(f"  Y offset: {transform_info['y_offset']}")
    print(f"  Resized size: {transform_info['resized_size']}")
    
    # Transform keypoint
    keypoints = [[[original_keypoint_x, original_keypoint_y, 1]]]  # One vertebra, one keypoint
    kp_transformer = KeypointTransformer()
    transformed_kps = kp_transformer.transform_keypoints(keypoints, transform_info)
    
    new_x = transformed_kps[0][0][0]
    new_y = transformed_kps[0][0][1]
    
    print(f"\nTransformed keypoint: ({new_x:.1f}, {new_y:.1f})")
    
    # Verify the transformation is correct
    # The keypoint should be at center of the resized (non-padded) portion
    expected_x = original_keypoint_x * transform_info['scale'] + transform_info['x_offset']
    expected_y = original_keypoint_y * transform_info['scale'] + transform_info['y_offset']
    
    print(f"Expected keypoint: ({expected_x:.1f}, {expected_y:.1f})")
    
    # Check if transformation is correct
    tolerance = 0.1
    if abs(new_x - expected_x) < tolerance and abs(new_y - expected_y) < tolerance:
        print("✓ PASS: Coordinates correctly scaled")
        return True
    else:
        print(f"✗ FAIL: Coordinate mismatch!")
        print(f"  Error X: {abs(new_x - expected_x):.4f}")
        print(f"  Error Y: {abs(new_y - expected_y):.4f}")
        return False


def test_heatmap_generation():
    """Test that heatmaps are generated at correct locations."""
    print("\n" + "="*60)
    print("TEST 2: Heatmap Generation at Correct Location")
    print("="*60)
    
    image_size = (512, 512)
    
    # Test keypoint at known location
    test_x, test_y = 256.0, 128.0  # Center-left
    keypoint = np.array([test_x, test_y, 1.0])
    
    print(f"Test keypoint: ({test_x}, {test_y})")
    
    # Generate heatmap
    heatmap = generate_gaussian_heatmap(
        height=image_size[1],
        width=image_size[0],
        keypoint=keypoint,
        sigma=3.0
    )
    
    # Find peak in heatmap
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_value = heatmap[peak_y, peak_x]
    
    print(f"Heatmap peak location: ({peak_x}, {peak_y})")
    print(f"Heatmap peak value: {peak_value:.4f}")
    
    # Check if peak is at the keypoint location (within 1 pixel)
    if abs(peak_x - test_x) <= 1 and abs(peak_y - test_y) <= 1:
        print("✓ PASS: Heatmap peak at correct location")
        return True
    else:
        print(f"✗ FAIL: Heatmap peak mismatch!")
        print(f"  Error X: {abs(peak_x - test_x):.1f} pixels")
        print(f"  Error Y: {abs(peak_y - test_y):.1f} pixels")
        return False


def test_dataset_coordinate_consistency():
    """Test that dataset returns consistent coordinates."""
    print("\n" + "="*60)
    print("TEST 3: Dataset Coordinate Consistency")
    print("="*60)
    
    # Load one sample from the dataset
    config = MacConfig
    
    try:
        dataset = UNetSpondylolisthesisDataset(
            image_dir=config.TRAIN_IMAGE_DIR,
            label_dir=config.TRAIN_LABEL_DIR,
            mode='val',  # No augmentation
            heatmap_sigma=config.HEATMAP_SIGMA,
            output_stride=config.OUTPUT_STRIDE,
            target_size=config.IMAGE_SIZE,
            apply_clahe=False
        )
        
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Get first sample
        sample = dataset[0]
        
        image = sample['image']
        heatmaps = sample['heatmaps']
        keypoints = sample['keypoints'].numpy()
        
        print(f"Image shape: {image.shape}")
        print(f"Heatmaps shape: {heatmaps.shape}")
        print(f"Keypoints shape: {keypoints.shape}")
        
        # Check first vertebra, first keypoint
        if len(keypoints) > 0:
            first_kp = keypoints[0][0]  # First vertebra, first corner
            kp_x, kp_y, kp_vis = first_kp
            
            print(f"\nFirst keypoint: ({kp_x:.1f}, {kp_y:.1f}, vis={kp_vis:.0f})")
            
            # Check if keypoint is within image bounds
            _, img_h, img_w = image.shape
            if 0 <= kp_x < img_w and 0 <= kp_y < img_h:
                print(f"✓ Keypoint within bounds [0, 0] to [{img_w}, {img_h}]")
                
                # Check corresponding heatmap
                heatmap_0 = heatmaps[0].numpy()  # First keypoint type
                
                # Find peak near the keypoint
                search_radius = 10
                y_min = max(0, int(kp_y) - search_radius)
                y_max = min(heatmap_0.shape[0], int(kp_y) + search_radius)
                x_min = max(0, int(kp_x) - search_radius)
                x_max = min(heatmap_0.shape[1], int(kp_x) + search_radius)
                
                region = heatmap_0[y_min:y_max, x_min:x_max]
                max_val = np.max(region)
                
                print(f"Heatmap max value near keypoint: {max_val:.4f}")
                
                if max_val > 0.5:  # Should have strong activation
                    print("✓ PASS: Heatmap has strong activation near keypoint")
                    return True
                else:
                    print(f"✗ FAIL: Weak heatmap activation ({max_val:.4f})")
                    print("  Coordinates may not be properly scaled!")
                    return False
            else:
                print(f"✗ FAIL: Keypoint out of bounds!")
                print(f"  Image size: ({img_w}, {img_h})")
                print(f"  Keypoint: ({kp_x:.1f}, {kp_y:.1f})")
                return False
        else:
            print("⚠ WARNING: No keypoints in sample")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_sample_with_heatmaps():
    """Visualize a sample to check coordinate alignment."""
    print("\n" + "="*60)
    print("TEST 4: Visual Verification (Saving Figure)")
    print("="*60)
    
    config = MacConfig
    
    try:
        dataset = UNetSpondylolisthesisDataset(
            image_dir=config.TRAIN_IMAGE_DIR,
            label_dir=config.TRAIN_LABEL_DIR,
            mode='val',
            heatmap_sigma=config.HEATMAP_SIGMA,
            output_stride=config.OUTPUT_STRIDE,
            target_size=config.IMAGE_SIZE,
            apply_clahe=False
        )
        
        # Get a sample
        sample = dataset[0]
        
        image = sample['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
        heatmaps = sample['heatmaps'].numpy()
        keypoints = sample['keypoints'].numpy()
        
        # Denormalize image for display
        image = np.clip(image, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Coordinate Scaling Verification', fontsize=16)
        
        # Plot image with keypoints
        ax = axes[0, 0]
        ax.imshow(image, cmap='gray')
        
        # Plot all keypoints
        colors = ['red', 'green', 'blue', 'yellow']
        for vertebra_kps in keypoints:
            for kp_idx, (x, y, vis) in enumerate(vertebra_kps):
                if vis > 0:
                    ax.plot(x, y, 'o', color=colors[kp_idx % 4], markersize=8)
        
        ax.set_title('Image with Keypoints')
        ax.axis('off')
        
        # Plot individual heatmaps
        heatmap_titles = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
        for i in range(4):
            row = (i + 1) // 3
            col = (i + 1) % 3
            ax = axes[row, col]
            
            heatmap = heatmaps[i]
            im = ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Heatmap: {heatmap_titles[i]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot overlay
        ax = axes[0, 1]
        ax.imshow(image, cmap='gray', alpha=0.7)
        
        # Overlay all heatmaps
        combined_heatmap = np.max(heatmaps, axis=0)
        ax.imshow(combined_heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.set_title('Image + Combined Heatmaps')
        ax.axis('off')
        
        # Save figure
        output_path = PROJECT_ROOT / 'tests' / 'coordinate_test_visualization.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {output_path}")
        print("  Open this file to visually verify keypoints align with heatmaps")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all coordinate scaling tests."""
    print("\n" + "="*80)
    print("COORDINATE SCALING TEST SUITE")
    print("Verifying that keypoints are correctly scaled when images are resized")
    print("="*80)
    
    results = []
    
    # Test 1: Basic coordinate transformation
    results.append(("Coordinate Scaling", test_coordinate_scaling()))
    
    # Test 2: Heatmap generation
    results.append(("Heatmap Generation", test_heatmap_generation()))
    
    # Test 3: Dataset consistency
    results.append(("Dataset Consistency", test_dataset_coordinate_consistency()))
    
    # Test 4: Visual verification
    results.append(("Visual Verification", visualize_sample_with_heatmaps()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Coordinate scaling is working correctly!")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("There may be issues with coordinate scaling!")
        return 1


if __name__ == '__main__':
    exit(main())
