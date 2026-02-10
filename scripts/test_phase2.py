#!/usr/bin/env python3
"""
Test Script for MAHT-Net Phase 2 and Phase 3 Components

Tests:
1. Transformer Bridge (Phase 2)
2. Vertebral Attention Module (VAM) (Phase 2)
3. Anatomical Structure Loss (Phase 2)
4. Combined MAHTNet Loss (Phase 2)
5. Full Phase 2 MAHT-Net model
6. LA view
7. DARK Decoding (Phase 3)
8. Uncertainty Estimation (Phase 3)
9. Full Phase 3 MAHT-Net model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

def test_transformer_bridge():
    """Test the Transformer Bridge component."""
    print("\n" + "="*60)
    print("1. Testing Transformer Bridge")
    print("="*60)
    
    from models.components.transformer_bridge import TransformerBridge
    
    bridge = TransformerBridge(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        feature_size=16
    )
    
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        out = bridge(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {out.shape}"
    print("✓ Transformer Bridge test passed!")
    return True


def test_vam():
    """Test the Vertebral Attention Module."""
    print("\n" + "="*60)
    print("2. Testing Vertebral Attention Module (VAM)")
    print("="*60)
    
    from models.components.vam import VertebralAttentionModule
    
    vam = VertebralAttentionModule(
        d_model=256,
        num_keypoints=20,
        nhead=8,
        num_layers=3,
        view='AP'
    )
    
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        out = vam(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == (2, 20, 256), f"Expected (2, 20, 256), got {out.shape}"
    print("✓ VAM test passed!")
    return True


def test_anatomical_loss():
    """Test the Anatomical Structure Loss."""
    print("\n" + "="*60)
    print("3. Testing Anatomical Structure Loss")
    print("="*60)
    
    from models.losses.anatomical_loss import AnatomicalStructureLoss
    
    loss_fn = AnatomicalStructureLoss(
        ordering_weight=1.0,
        parallelism_weight=0.5,
        ratio_weight=0.3
    )
    
    # Simulated keypoints - slightly perturbed from a typical vertebral arrangement
    pred = torch.zeros(2, 20, 2)
    
    # Create realistic vertebral keypoint positions
    for b in range(2):
        for v in range(5):  # 5 vertebrae
            y_base = 100 + v * 80  # Vertical position
            for i, (dx, dy) in enumerate([(0, 0), (50, 0), (0, 60), (50, 60)]):
                idx = v * 4 + i
                pred[b, idx] = torch.tensor([100 + dx, y_base + dy], dtype=torch.float)
                # Add small noise
                pred[b, idx] += torch.randn(2) * 5
    
    loss, components = loss_fn(pred, view='AP')
    
    print(f"Predicted shape: {pred.shape}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Components:")
    for name, value in components.items():
        print(f"  {name}: {value.item():.4f}")
    
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"
    print("✓ Anatomical Structure Loss test passed!")
    return True


def test_combined_loss():
    """Test the Combined MAHTNet Loss."""
    print("\n" + "="*60)
    print("4. Testing Combined MAHTNet Loss")
    print("="*60)
    
    from models.losses.combined_loss import MAHTNetLoss
    
    loss_fn = MAHTNetLoss(
        heatmap_loss_type='mse',
        coord_loss_type='wing',
        use_anatomical_loss=True,
        lambda_anatomical=0.1,
        view='AP'
    )
    
    # Create dummy predictions and ground truth
    pred_heatmaps = torch.randn(2, 20, 512, 512) * 0.1
    gt_heatmaps = torch.randn(2, 20, 512, 512) * 0.1
    
    # Add some structure - peaks in heatmaps
    for b in range(2):
        for k in range(20):
            y, x = 100 + k * 20, 100 + (k % 4) * 50
            y = min(y, 500)
            gt_heatmaps[b, k, y-5:y+5, x-5:x+5] = 1.0
            pred_heatmaps[b, k, y-4:y+6, x-4:x+6] = 0.9
    
    loss, components = loss_fn(pred_heatmaps, gt_heatmaps)
    
    print(f"Predicted heatmaps: {pred_heatmaps.shape}")
    print(f"Ground truth heatmaps: {gt_heatmaps.shape}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Components:")
    for name, value in components.items():
        val = value.item() if torch.is_tensor(value) else value
        print(f"  {name}: {val:.4f}")
    
    assert loss.ndim == 0, "Loss should be scalar"
    print("✓ Combined MAHTNet Loss test passed!")
    return True


def test_full_phase2_model():
    """Test the Full Phase 2 MAHT-Net model."""
    print("\n" + "="*60)
    print("5. Testing Full Phase 2 MAHT-Net Model")
    print("="*60)
    
    from models.maht_net import create_maht_net
    
    # Create Phase 2 model
    model = create_maht_net(view='AP', phase=2)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = model(x)
    
    heatmaps = output['heatmaps']
    vam_features = output.get('vam_features')
    
    print(f"Input shape: {x.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    print(f"VAM features: {vam_features.shape if vam_features is not None else 'N/A'}")
    
    assert heatmaps.shape == (2, 20, 512, 512), f"Expected (2, 20, 512, 512), got {heatmaps.shape}"
    assert vam_features is not None, "VAM features should be present in Phase 2"
    assert vam_features.shape == (2, 20, 256), f"Expected (2, 20, 256), got {vam_features.shape}"
    
    # Test keypoint extraction
    keypoints = model.extract_keypoints(heatmaps)
    print(f"Extracted keypoints: {keypoints.shape}")
    assert keypoints.shape == (2, 20, 2)
    
    print("✓ Full Phase 2 MAHT-Net model test passed!")
    return True


def test_la_view():
    """Test LA view (22 keypoints)."""
    print("\n" + "="*60)
    print("6. Testing LA View (22 keypoints)")
    print("="*60)
    
    from models.maht_net import create_maht_net
    
    model = create_maht_net(view='LA', phase=2)
    
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"LA view heatmaps: {output['heatmaps'].shape}")
    print(f"LA view VAM features: {output['vam_features'].shape}")
    
    assert output['heatmaps'].shape == (2, 22, 512, 512)
    assert output['vam_features'].shape == (2, 22, 256)
    
    print("✓ LA View test passed!")
    return True


def test_dark_decoding():
    """Test DARK decoding for sub-pixel coordinates."""
    print("\n" + "="*60)
    print("7. Testing DARK Decoding (Phase 3)")
    print("="*60)
    
    from src.utils.dark_decoding import dark_decoding, soft_argmax
    
    B, K, H, W = 2, 20, 512, 512
    
    # Create test heatmaps with Gaussian peaks
    heatmaps = torch.zeros(B, K, H, W)
    
    for b in range(B):
        for k in range(K):
            x_true = 100.3 + k * 20
            y_true = 150.7 + k * 15
            
            if x_true < W and y_true < H:
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, dtype=torch.float32),
                    torch.arange(W, dtype=torch.float32),
                    indexing='ij'
                )
                sigma = 2.0
                gaussian = torch.exp(-((x_grid - x_true)**2 + (y_grid - y_true)**2) / (2 * sigma**2))
                heatmaps[b, k] = gaussian
    
    # Test DARK decoding
    coords_dark, confidence = dark_decoding(heatmaps)
    
    # Test soft-argmax
    coords_soft = soft_argmax(heatmaps)
    
    print(f"Input heatmaps: {heatmaps.shape}")
    print(f"DARK coords: {coords_dark.shape}")
    print(f"Soft-argmax coords: {coords_soft.shape}")
    print(f"Confidence: {confidence.shape}")
    print(f"Sample DARK coord (kp 0): ({coords_dark[0, 0, 0]:.2f}, {coords_dark[0, 0, 1]:.2f})")
    print(f"True location: (100.30, 150.70)")
    
    assert coords_dark.shape == (B, K, 2)
    assert confidence.shape == (B, K)
    
    print("✓ DARK Decoding test passed!")
    return True


def test_uncertainty_estimation():
    """Test uncertainty estimation modules."""
    print("\n" + "="*60)
    print("8. Testing Uncertainty Estimation (Phase 3)")
    print("="*60)
    
    from models.components.uncertainty import (
        UncertaintyHead, 
        HeatmapSpreadUncertainty,
        NLLLoss
    )
    
    B, K, H, W = 2, 20, 512, 512
    d_model = 256
    
    # Test learned uncertainty head
    vam_features = torch.randn(B, K, d_model)
    head = UncertaintyHead(d_model, K)
    sigma_learned = head(vam_features)
    
    print(f"VAM features: {vam_features.shape}")
    print(f"Learned uncertainty: {sigma_learned.shape}")
    print(f"Sigma range: [{sigma_learned.min():.2f}, {sigma_learned.max():.2f}]")
    
    assert sigma_learned.shape == (B, K, 2)
    
    # Test heatmap-based uncertainty
    heatmaps = torch.randn(B, K, H, W)
    spread = HeatmapSpreadUncertainty()
    sigma_spread = spread(heatmaps)
    
    print(f"Heatmap uncertainty: {sigma_spread.shape}")
    
    assert sigma_spread.shape == (B, K, 2)
    
    # Test NLL loss
    nll_loss = NLLLoss()
    pred_coords = torch.randn(B, K, 2) * 50 + 256
    gt_coords = pred_coords + torch.randn(B, K, 2) * 5
    sigma = torch.ones(B, K, 2) * 5.0
    
    loss = nll_loss(pred_coords, gt_coords, sigma)
    print(f"NLL Loss: {loss.item():.4f}")
    
    print("✓ Uncertainty Estimation test passed!")
    return True


def test_full_phase3_model():
    """Test the Full Phase 3 MAHT-Net model with uncertainty."""
    print("\n" + "="*60)
    print("9. Testing Full Phase 3 MAHT-Net Model")
    print("="*60)
    
    from models.maht_net import create_maht_net
    
    # Create Phase 3 model
    model = create_maht_net(view='AP', phase=3)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = model(x)
    
    heatmaps = output['heatmaps']
    vam_features = output.get('vam_features')
    uncertainty = output.get('uncertainty')
    
    print(f"Input shape: {x.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    print(f"VAM features: {vam_features.shape if vam_features is not None else 'N/A'}")
    print(f"Uncertainty: {uncertainty.shape if uncertainty is not None else 'N/A'}")
    
    assert heatmaps.shape == (2, 20, 512, 512)
    assert vam_features is not None and vam_features.shape == (2, 20, 256)
    assert uncertainty is not None and uncertainty.shape == (2, 20, 2)
    
    # Test DARK decoding via model method
    keypoints_dark, confidence = model.extract_keypoints_dark(heatmaps)
    print(f"DARK keypoints: {keypoints_dark.shape}")
    print(f"Confidence: {confidence.shape}")
    
    assert keypoints_dark.shape == (2, 20, 2)
    assert confidence.shape == (2, 20)
    
    print("✓ Full Phase 3 MAHT-Net model test passed!")
    return True


def run_all_tests():
    """Run all Phase 2 and Phase 3 tests."""
    print("\n" + "#"*60)
    print("# MAHT-Net Phase 2 & Phase 3 Component Tests")
    print("#"*60)
    
    tests = [
        # Phase 2 tests
        test_transformer_bridge,
        test_vam,
        test_anatomical_loss,
        test_combined_loss,
        test_full_phase2_model,
        test_la_view,
        # Phase 3 tests
        test_dark_decoding,
        test_uncertainty_estimation,
        test_full_phase3_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*60)
    print(f"# Results: {passed} passed, {failed} failed")
    print("#"*60)
    
    if failed == 0:
        print("\n🎉 All Phase 2 & Phase 3 tests passed! Ready for training.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
