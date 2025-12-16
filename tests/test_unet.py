"""
Testing script to verify UNet implementation before running on Google Colab.
Tests all components: model, dataset, training loop.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.unet import create_unet
from src.data.unet_dataset import UNetSpondylolisthesisDataset, create_unet_dataloaders
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import SpondylolisthesisAugmentation
from scripts.train_unet import UNetKeypointLoss
from evaluation.unet_metrics import evaluate_unet_predictions


def test_model_creation():
    """Test 1: Model creation and forward pass."""
    print("\n" + "="*60)
    print("TEST 1: Model Creation and Forward Pass")
    print("="*60)
    
    try:
        # Create model
        model = create_unet(in_channels=3, num_keypoints=4, bilinear=False, base_channels=64)
        print("✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 512, 512)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        expected_shape = (batch_size, 4, 512, 512)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Output shape correct: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test 2: Dataset loading and data generation."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Loading and Data Generation")
    print("="*60)
    
    try:
        # Data paths
        data_root = project_root / 'data' / 'Train' / 'Keypointrcnn_data'
        train_image_dir = data_root / 'images' / 'train'
        train_label_dir = data_root / 'labels' / 'train'
        
        # Check if data exists
        if not train_image_dir.exists() or not train_label_dir.exists():
            print(f"✗ Data directories not found:")
            print(f"  {train_image_dir}")
            print(f"  {train_label_dir}")
            return False
        
        print(f"✓ Data directories found")
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(512, 512),
            normalize=True,
            apply_clahe=True
        )
        print("✓ Preprocessor created")
        
        # Create dataset
        dataset = UNetSpondylolisthesisDataset(
            image_dir=train_image_dir,
            label_dir=train_label_dir,
            mode='train',
            preprocessor=preprocessor,
            heatmap_sigma=3.0,
            num_keypoint_types=4,
            output_stride=1
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Get a sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape:    {sample['image'].shape}")
        print(f"  Heatmap shape:  {sample['heatmaps'].shape}")
        print(f"  Keypoints shape: {sample['keypoints'].shape}")
        print(f"  Filename:       {sample['filename']}")
        
        # Check shapes
        assert sample['image'].shape[0] == 3, "Image should have 3 channels"
        assert sample['heatmaps'].shape[0] == 4, "Should have 4 heatmaps (corners)"
        assert sample['heatmaps'].shape[1:] == sample['image'].shape[1:], "Heatmap and image sizes should match"
        print("✓ All shapes correct")
        
        # Check heatmap values
        heatmap_min = sample['heatmaps'].min().item()
        heatmap_max = sample['heatmaps'].max().item()
        print(f"✓ Heatmap value range: [{heatmap_min:.3f}, {heatmap_max:.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test 3: DataLoader and batching."""
    print("\n" + "="*60)
    print("TEST 3: DataLoader and Batching")
    print("="*60)
    
    try:
        # Data paths
        data_root = project_root / 'data' / 'Train' / 'Keypointrcnn_data'
        train_image_dir = data_root / 'images' / 'train'
        train_label_dir = data_root / 'labels' / 'train'
        val_image_dir = data_root / 'images' / 'val'
        val_label_dir = data_root / 'labels' / 'val'
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(512, 512),
            normalize=True,
            apply_clahe=True
        )
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader = create_unet_dataloaders(
            train_image_dir=train_image_dir,
            train_label_dir=train_label_dir,
            val_image_dir=val_image_dir,
            val_label_dir=val_label_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            heatmap_sigma=3.0,
            output_stride=1,
            preprocessor=preprocessor
        )
        print(f"✓ DataLoaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"  Images shape:   {batch['images'].shape}")
        print(f"  Heatmaps shape: {batch['heatmaps'].shape}")
        print(f"  Batch size:     {len(batch['filenames'])}")
        
        # Check batch
        assert batch['images'].shape[0] == 4, "Batch size should be 4"
        assert batch['heatmaps'].shape[0] == 4, "Batch size should be 4"
        assert len(batch['keypoints']) == 4, "Should have keypoints for 4 images"
        print("✓ Batch structure correct")
        
        return True
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """Test 4: Loss function."""
    print("\n" + "="*60)
    print("TEST 4: Loss Function")
    print("="*60)
    
    try:
        # Create loss function
        criterion = UNetKeypointLoss(use_focal=True, focal_alpha=2.0, focal_beta=4.0)
        print("✓ Loss function created")
        
        # Create dummy predictions and targets
        batch_size = 2
        pred_heatmaps = torch.randn(batch_size, 4, 128, 128, requires_grad=True)
        target_heatmaps = torch.rand(batch_size, 4, 128, 128)
        
        # Compute loss
        loss = criterion(pred_heatmaps, target_heatmaps)
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        # Check loss properties
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
        print("✓ Loss value valid")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test 5: Training loop (1 epoch, few batches)."""
    print("\n" + "="*60)
    print("TEST 5: Training Loop (Mini)")
    print("="*60)
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Create model
        model = create_unet(in_channels=3, num_keypoints=4, bilinear=False, base_channels=32)  # Smaller for testing
        model = model.to(device)
        print("✓ Model created and moved to device")
        
        # Data paths
        data_root = project_root / 'data' / 'Train' / 'Keypointrcnn_data'
        train_image_dir = data_root / 'images' / 'train'
        train_label_dir = data_root / 'labels' / 'train'
        val_image_dir = data_root / 'images' / 'val'
        val_label_dir = data_root / 'labels' / 'val'
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(256, 256),  # Smaller for testing
            normalize=True,
            apply_clahe=True
        )
        
        # Create dataloaders
        train_loader, val_loader = create_unet_dataloaders(
            train_image_dir=train_image_dir,
            train_label_dir=train_label_dir,
            val_image_dir=val_image_dir,
            val_label_dir=val_label_dir,
            batch_size=2,
            num_workers=0,
            heatmap_sigma=3.0,
            output_stride=1,
            preprocessor=preprocessor
        )
        print("✓ DataLoaders created")
        
        # Create loss and optimizer
        criterion = UNetKeypointLoss(use_focal=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("✓ Loss and optimizer created")
        
        # Train for 2 batches
        model.train()
        print("\nTraining for 2 batches...")
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            
            images = batch['images'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"  Batch {i+1}: Loss = {loss.item():.4f}")
        
        print("✓ Training loop successful")
        
        # Test validation
        model.eval()
        print("\nValidating for 1 batch...")
        with torch.no_grad():
            batch = next(iter(val_loader))
            images = batch['images'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps)
            
            print(f"  Val Loss = {loss.item():.4f}")
        
        print("✓ Validation loop successful")
        
        return True
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test 6: Evaluation metrics."""
    print("\n" + "="*60)
    print("TEST 6: Evaluation Metrics")
    print("="*60)
    
    try:
        # Create dummy predictions and targets
        batch_size = 2
        pred_heatmaps = torch.randn(batch_size, 4, 128, 128)
        target_heatmaps = torch.rand(batch_size, 4, 128, 128)
        
        # Create dummy keypoints
        target_keypoints = [
            torch.rand(3, 4, 3) * 128,  # 3 vertebrae, 4 corners, [x,y,vis]
            torch.rand(2, 4, 3) * 128   # 2 vertebrae, 4 corners, [x,y,vis]
        ]
        
        # Evaluate
        metrics = evaluate_unet_predictions(
            pred_heatmaps,
            target_heatmaps,
            target_keypoints,
            threshold=0.5,
            image_size=(128, 128)
        )
        
        print("✓ Metrics computed successfully")
        print("\nMetric values:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Check metrics
        assert 'heatmap_mse' in metrics, "Should have heatmap_mse"
        assert 'pck_0.05' in metrics, "Should have PCK metrics"
        print("✓ All expected metrics present")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("UNET IMPLEMENTATION VERIFICATION")
    print("="*60)
    print("\nRunning comprehensive tests before Colab deployment...\n")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset_loading),
        ("DataLoader", test_dataloader),
        ("Loss Function", test_loss_function),
        ("Training Loop", test_training_loop),
        ("Metrics", test_metrics),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for Google Colab deployment.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix issues before Colab deployment.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
