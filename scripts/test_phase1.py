#!/usr/bin/env python3
"""
Test script for MAHT-Net Phase 1 components.

Tests:
1. BUU-LSPINE dataset loading
2. Heatmap generation
3. EfficientNet backbone
4. Simple decoder
5. Full MAHT-Net forward pass
6. End-to-end training step
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt


def test_dataset():
    """Test BUU-LSPINE dataset loading."""
    print("\n" + "="*60)
    print("1. Testing BUU-LSPINE Dataset Loader")
    print("="*60)
    
    from src.data.buu_lspine_dataset import BUULSpineDataset
    
    root_dir = PROJECT_ROOT / "data" / "buu-lspine"
    
    # Test AP view dataset
    print("\nLoading AP view dataset...")
    dataset = BUULSpineDataset(
        root_dir=str(root_dir),
        view='AP',
        split='train',
        image_size=(512, 512),
        sigma=4.0
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample contents:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Keypoints shape: {sample['keypoints'].shape}")
    print(f"  Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"  Spondy labels shape: {sample['spondy_labels'].shape}")
    print(f"  Pixel spacing: {sample['pixel_spacing'].item():.4f} mm/px")
    
    # Verify shapes
    assert sample['image'].shape == (3, 512, 512), f"Wrong image shape"
    assert sample['keypoints'].shape == (20, 2), f"Wrong keypoints shape"
    assert sample['heatmaps'].shape == (20, 512, 512), f"Wrong heatmaps shape"
    
    # Test LA view
    print("\nLoading LA view dataset...")
    la_dataset = BUULSpineDataset(
        root_dir=str(root_dir),
        view='LA',
        split='train'
    )
    la_sample = la_dataset[0]
    assert la_sample['keypoints'].shape == (22, 2), "LA should have 22 keypoints"
    print(f"LA dataset size: {len(la_dataset)}, keypoints: {la_sample['keypoints'].shape}")
    
    print("\n✓ Dataset test passed!")
    return sample


def test_backbone():
    """Test EfficientNetV2 backbone."""
    print("\n" + "="*60)
    print("2. Testing EfficientNetV2 Backbone")
    print("="*60)
    
    from models.components.efficientnet_backbone import EfficientNetV2Backbone
    
    backbone = EfficientNetV2Backbone(pretrained=True, freeze_stages=2)
    
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        f4, skips = backbone(x)
    
    print(f"\nInput: {x.shape}")
    print(f"F4 output: {f4.shape}")
    print(f"Skip features: {[s.shape for s in skips]}")
    
    # Verify output shapes
    assert f4.shape == (2, 256, 16, 16), f"Wrong F4 shape: {f4.shape}"
    
    print("\n✓ Backbone test passed!")
    return backbone


def test_decoder():
    """Test simple decoder."""
    print("\n" + "="*60)
    print("3. Testing Simple Decoder")
    print("="*60)
    
    from models.components.decoder import SimpleDecoder
    
    decoder = SimpleDecoder(num_keypoints=20, in_channels=256)
    
    # Simulated backbone outputs
    f4 = torch.randn(2, 256, 16, 16)
    f1 = torch.randn(2, 48, 128, 128)
    f2 = torch.randn(2, 64, 64, 64)
    f3 = torch.randn(2, 128, 32, 32)
    
    with torch.no_grad():
        heatmaps = decoder(f4, [f1, f2, f3])
    
    print(f"\nF4 input: {f4.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    
    assert heatmaps.shape == (2, 20, 512, 512), f"Wrong output shape: {heatmaps.shape}"
    
    print("\n✓ Decoder test passed!")
    return decoder


def test_maht_net():
    """Test full MAHT-Net model."""
    print("\n" + "="*60)
    print("4. Testing Full MAHT-Net (Phase 1)")
    print("="*60)
    
    from models.maht_net import MAHTNet
    
    model = MAHTNet(num_keypoints=20, pretrained_backbone=True)
    
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
    
    heatmaps = output['heatmaps']
    print(f"\nInput: {x.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    
    # Test keypoint extraction
    keypoints = model.extract_keypoints(heatmaps)
    print(f"Extracted keypoints: {keypoints.shape}")
    
    assert heatmaps.shape == (2, 20, 512, 512)
    assert keypoints.shape == (2, 20, 2)
    
    print("\n✓ MAHT-Net test passed!")
    return model


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("5. Testing Training Step")
    print("="*60)
    
    from models.maht_net import MAHTNet
    from src.data.buu_lspine_dataset import BUULSpineDataset
    from torch.utils.data import DataLoader
    
    # Create model
    model = MAHTNet(num_keypoints=20, pretrained_backbone=True)
    
    # Create dataset and loader
    root_dir = PROJECT_ROOT / "data" / "buu-lspine"
    dataset = BUULSpineDataset(
        root_dir=str(root_dir),
        view='AP',
        split='train',
        image_size=(512, 512)
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a batch
    batch = next(iter(loader))
    images = batch['image']
    gt_heatmaps = batch['heatmaps']
    
    print(f"\nBatch images: {images.shape}")
    print(f"Batch GT heatmaps: {gt_heatmaps.shape}")
    
    # Forward pass
    model.train()
    output = model(images)
    pred_heatmaps = output['heatmaps']
    
    # Compute loss
    loss = model.compute_loss(pred_heatmaps, gt_heatmaps)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters() if p.requires_grad)
    print(f"Gradients computed: {has_grad}")
    
    assert has_grad, "No gradients computed!"
    
    print("\n✓ Training step test passed!")
    return True


def visualize_sample(sample, save_path=None):
    """Visualize a dataset sample."""
    print("\n" + "="*60)
    print("6. Visualizing Sample")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Denormalize image
    img = sample['image'].numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    keypoints = sample['keypoints'].numpy()
    heatmap_sum = sample['heatmaps'].sum(dim=0).numpy()
    
    # Image with keypoints
    axes[0].imshow(img)
    axes[0].scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=30, marker='o')
    axes[0].set_title('Image with Keypoints')
    axes[0].axis('off')
    
    # Heatmap sum
    axes[1].imshow(heatmap_sum, cmap='hot')
    axes[1].set_title('Summed Heatmaps')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(heatmap_sum, alpha=0.5, cmap='hot')
    axes[2].scatter(keypoints[:, 0], keypoints[:, 1], c='cyan', s=20, marker='x')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    print("\n✓ Visualization complete!")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# MAHT-Net Phase 1 Component Tests")
    print("#"*60)
    
    # Run tests
    sample = test_dataset()
    test_backbone()
    test_decoder()
    test_maht_net()
    test_training_step()
    
    # Save visualization
    vis_path = PROJECT_ROOT / "experiments" / "visualizations"
    vis_path.mkdir(parents=True, exist_ok=True)
    visualize_sample(sample, vis_path / "phase1_test_sample.png")
    
    print("\n" + "#"*60)
    print("# All Phase 1 tests passed! ✓")
    print("#"*60)
    print("""
Next steps:
1. Run: python scripts/test_phase1.py
2. Check visualization: experiments/visualizations/phase1_test_sample.png
3. Start training: python train.py --model maht_net --view AP
""")


if __name__ == "__main__":
    main()
