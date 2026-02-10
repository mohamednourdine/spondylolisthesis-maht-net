"""
MAHT-Net: Multi-scale Anatomical Heatmap Transformer

Implementation for vertebral corner point detection in lumbar spine X-rays.

Phases:
- Phase 1: Backbone + Simple Decoder (baseline)
- Phase 2: + Transformer Bridge + Vertebral Attention Module (VAM)
- Phase 3: + DARK decoding + uncertainty estimation (future)

The architecture can be configured to run in different phases for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .components.efficientnet_backbone import EfficientNetV2Backbone
from .components.decoder import SimpleDecoder, MultiScaleDecoder
from .components.transformer_bridge import TransformerBridge
from .components.vam import VertebralAttentionModule
from .components.uncertainty import UncertaintyHead, HeatmapSpreadUncertainty


class MAHTNet(nn.Module):
    """
    MAHT-Net: Multi-scale Anatomical Heatmap Transformer
    
    A deep learning architecture for precise vertebral corner point
    detection in lumbar spine X-ray images using heatmap regression.
    
    Architecture phases:
    - Phase 1: Backbone → Decoder (simple baseline)
    - Phase 2: Backbone → Transformer → VAM → Decoder (full architecture)
    
    Args:
        num_keypoints: Number of keypoints (20 for AP, 22 for LA view)
        d_model: Feature dimension (256)
        pretrained_backbone: Use ImageNet pretrained weights
        freeze_backbone_stages: Number of backbone stages to freeze (0-4)
        decoder_channels: Base decoder channel dimension
        dropout: Dropout rate
        use_transformer: Enable Transformer Bridge (Phase 2)
        use_vam: Enable Vertebral Attention Module (Phase 2)
        use_uncertainty: Enable uncertainty estimation (Phase 3)
        transformer_layers: Number of transformer encoder layers
        vam_layers: Number of VAM attention layers
        view: 'AP' or 'LA' for view-specific configurations
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        d_model: int = 256,
        pretrained_backbone: bool = True,
        freeze_backbone_stages: int = 2,
        decoder_channels: int = 128,
        dropout: float = 0.1,
        use_transformer: bool = False,
        use_vam: bool = False,
        use_uncertainty: bool = False,
        transformer_layers: int = 4,
        vam_layers: int = 3,
        view: str = 'AP'
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.use_transformer = use_transformer
        self.use_vam = use_vam
        self.use_uncertainty = use_uncertainty
        self.view = view
        
        # Determine phase
        if use_uncertainty and use_transformer and use_vam:
            phase = "Phase 3: MAHT-Net + Uncertainty"
        elif use_transformer and use_vam:
            phase = "Phase 2: Full MAHT-Net"
        elif use_transformer:
            phase = "Phase 1.5: Backbone + Transformer + Decoder"
        else:
            phase = "Phase 1: Backbone + Decoder"
        
        print(f"\n{'='*60}")
        print(f"Initializing MAHT-Net ({phase})")
        print(f"{'='*60}")
        
        # 1. CNN Backbone - EfficientNetV2-S with multi-scale outputs
        self.backbone = EfficientNetV2Backbone(
            pretrained=pretrained_backbone,
            freeze_stages=freeze_backbone_stages,
            out_channels=d_model
        )
        
        # Get skip channel dimensions from backbone
        feature_channels = self.backbone.get_feature_channels()
        skip_channels = [
            feature_channels['F1'],  # 48
            feature_channels['F2'],  # 64
            feature_channels['F3'],  # 128
        ]
        
        # 2. Transformer Bridge (Phase 2)
        if use_transformer:
            self.transformer = TransformerBridge(
                d_model=d_model,
                nhead=8,
                num_layers=transformer_layers,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                feature_size=16
            )
        else:
            self.transformer = None
        
        # 3. Vertebral Attention Module (Phase 2)
        if use_vam:
            self.vam = VertebralAttentionModule(
                d_model=d_model,
                num_keypoints=num_keypoints,
                nhead=8,
                num_layers=vam_layers,
                dropout=dropout,
                view=view
            )
            # Use MultiScaleDecoder when VAM is enabled
            self.decoder = MultiScaleDecoder(
                num_keypoints=num_keypoints,
                d_model=d_model,
                skip_channels=skip_channels,
                decoder_channels=decoder_channels,
                dropout=dropout
            )
        else:
            self.vam = None
            # Use SimpleDecoder for Phase 1
            self.decoder = SimpleDecoder(
                num_keypoints=num_keypoints,
                in_channels=d_model,
                skip_channels=skip_channels,
                decoder_channels=decoder_channels,
                dropout=dropout
            )
        
        # 4. Uncertainty Head (Phase 3)
        if use_uncertainty:
            if use_vam:
                # Learned uncertainty from VAM features
                self.uncertainty_head = UncertaintyHead(
                    d_model=d_model,
                    num_keypoints=num_keypoints,
                    dropout=dropout
                )
            else:
                # Heatmap-based uncertainty only
                self.uncertainty_head = None
            # Always include heatmap spread uncertainty
            self.heatmap_uncertainty = HeatmapSpreadUncertainty()
        else:
            self.uncertainty_head = None
            self.heatmap_uncertainty = None
        
        # Count parameters
        total_params = self._count_parameters()
        trainable_params = self._count_trainable_parameters()
        
        print(f"{'='*60}")
        print(f"MAHT-Net initialized")
        print(f"  Total parameters: {total_params:.2f}M")
        print(f"  Trainable parameters: {trainable_params:.2f}M")
        print(f"  Output keypoints: {num_keypoints}")
        print(f"  Transformer: {use_transformer}")
        print(f"  VAM: {use_vam}")
        print(f"  Uncertainty: {use_uncertainty}")
        print(f"{'='*60}\n")
    
    def _count_parameters(self) -> float:
        """Count total parameters in millions."""
        return sum(p.numel() for p in self.parameters()) / 1e6
    
    def _count_trainable_parameters(self) -> float:
        """Count trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MAHT-Net.
        
        Args:
            x: Input images (B, 3, 512, 512)
            
        Returns:
            dict with:
                'heatmaps': (B, K, 512, 512) - predicted heatmaps
                'vam_features': (B, K, C) - VAM features if enabled
                'uncertainty': (B, K, 2) - uncertainty (σ_x, σ_y) if enabled
        """
        output = {}
        
        # 1. Extract multi-scale features from backbone
        f4, skip_features = self.backbone(x)
        
        # 2. Apply Transformer Bridge (if enabled)
        if self.transformer is not None:
            global_features = self.transformer(f4)
        else:
            global_features = f4
        
        # 3. Apply VAM (if enabled)
        if self.vam is not None:
            vam_features = self.vam(global_features)
            output['vam_features'] = vam_features
            
            # Generate heatmaps via MultiScaleDecoder
            heatmaps = self.decoder(vam_features, skip_features, global_features)
        else:
            # Generate heatmaps via SimpleDecoder
            heatmaps = self.decoder(global_features, skip_features)
        
        output['heatmaps'] = heatmaps
        
        # 4. Estimate uncertainty (if enabled)
        if self.use_uncertainty:
            if self.uncertainty_head is not None and 'vam_features' in output:
                # Learned uncertainty from VAM features
                uncertainty = self.uncertainty_head(output['vam_features'])
            else:
                # Heatmap-based uncertainty
                uncertainty = self.heatmap_uncertainty(heatmaps)
            output['uncertainty'] = uncertainty
        
        return output
    
    def extract_keypoints(
        self, 
        heatmaps: torch.Tensor,
        use_softmax: bool = True
    ) -> torch.Tensor:
        """
        Extract keypoint coordinates from heatmaps.
        
        Uses soft-argmax for differentiable coordinate extraction.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            use_softmax: Apply softmax for soft-argmax (True) or hard argmax (False)
            
        Returns:
            keypoints: (B, K, 2) coordinates in pixel space
        """
        B, K, H, W = heatmaps.shape
        
        if use_softmax:
            # Soft-argmax: differentiable coordinate extraction
            # Flatten spatial dimensions
            heatmaps_flat = heatmaps.view(B, K, -1)
            
            # Apply softmax to get probability distribution
            probs = F.softmax(heatmaps_flat, dim=-1)
            
            # Create coordinate grids
            device = heatmaps.device
            x_coords = torch.arange(W, device=device).float()
            y_coords = torch.arange(H, device=device).float()
            
            # Expected x = sum(prob * x_coord)
            x_probs = probs.view(B, K, H, W).sum(dim=2)  # (B, K, W)
            y_probs = probs.view(B, K, H, W).sum(dim=3)  # (B, K, H)
            
            x = (x_probs * x_coords).sum(dim=-1)  # (B, K)
            y = (y_probs * y_coords).sum(dim=-1)  # (B, K)
            
        else:
            # Hard argmax (non-differentiable)
            heatmaps_flat = heatmaps.view(B, K, -1)
            max_idx = heatmaps_flat.argmax(dim=2)
            
            y = max_idx // W
            x = max_idx % W
            x = x.float()
            y = y.float()
        
        keypoints = torch.stack([x, y], dim=2)  # (B, K, 2)
        return keypoints
    
    def extract_keypoints_dark(
        self,
        heatmaps: torch.Tensor,
        kernel_size: int = 11
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract keypoint coordinates using DARK decoding.
        
        DARK (Distribution-Aware coordinate Representation of Keypoints)
        provides sub-pixel accurate coordinates by fitting a Taylor expansion
        around the heatmap peak.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            kernel_size: Gaussian smoothing kernel size
            
        Returns:
            keypoints: (B, K, 2) refined coordinates in pixel space
            confidence: (B, K) confidence scores
        """
        from src.utils.dark_decoding import dark_decoding
        return dark_decoding(heatmaps, kernel_size=kernel_size)
    
    def compute_loss(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute heatmap MSE loss.
        
        Args:
            pred_heatmaps: (B, K, H, W) predicted heatmaps
            gt_heatmaps: (B, K, H, W) ground truth heatmaps
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: Scalar loss value
        """
        return F.mse_loss(pred_heatmaps, gt_heatmaps, reduction=reduction)


def create_maht_net(
    view: str = 'AP',
    phase: int = 2,
    **kwargs
) -> MAHTNet:
    """
    Factory function to create MAHT-Net for specific view and phase.
    
    Args:
        view: 'AP' (20 keypoints) or 'LA' (22 keypoints)
        phase: 1 (baseline), 2 (Transformer + VAM), or 3 (+ uncertainty)
        **kwargs: Additional arguments for MAHTNet
        
    Returns:
        Configured MAHTNet instance
    """
    num_keypoints = 20 if view.upper() == 'AP' else 22
    
    # Configure based on phase
    if phase == 1:
        kwargs.setdefault('use_transformer', False)
        kwargs.setdefault('use_vam', False)
        kwargs.setdefault('use_uncertainty', False)
    elif phase == 2:
        kwargs.setdefault('use_transformer', True)
        kwargs.setdefault('use_vam', True)
        kwargs.setdefault('use_uncertainty', False)
    elif phase == 3:
        kwargs.setdefault('use_transformer', True)
        kwargs.setdefault('use_vam', True)
        kwargs.setdefault('use_uncertainty', True)
    
    return MAHTNet(
        num_keypoints=num_keypoints,
        view=view,
        **kwargs
    )


def test_maht_net():
    """Test the MAHT-Net model."""
    print("\n" + "#"*60)
    print("# Testing MAHT-Net")
    print("#"*60)
    
    # Test Phase 1
    print("\n1. Testing Phase 1 (Backbone + Decoder)...")
    model_p1 = create_maht_net(view='AP', phase=1)
    
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output_p1 = model_p1(x)
    
    heatmaps_p1 = output_p1['heatmaps']
    print(f"   Input shape: {x.shape}")
    print(f"   Output heatmaps: {heatmaps_p1.shape}")
    assert heatmaps_p1.shape == (2, 20, 512, 512)
    print("   ✓ Phase 1 passed")
    
    # Test Phase 2
    print("\n2. Testing Phase 2 (Full MAHT-Net with Transformer + VAM)...")
    model_p2 = create_maht_net(view='AP', phase=2)
    
    with torch.no_grad():
        output_p2 = model_p2(x)
    
    heatmaps_p2 = output_p2['heatmaps']
    vam_features = output_p2.get('vam_features')
    
    print(f"   Output heatmaps: {heatmaps_p2.shape}")
    print(f"   VAM features: {vam_features.shape if vam_features is not None else 'N/A'}")
    assert heatmaps_p2.shape == (2, 20, 512, 512)
    assert vam_features.shape == (2, 20, 256)
    print("   ✓ Phase 2 passed")
    
    # Test Phase 3
    print("\n3. Testing Phase 3 (MAHT-Net + Uncertainty)...")
    model_p3 = create_maht_net(view='AP', phase=3)
    
    with torch.no_grad():
        output_p3 = model_p3(x)
    
    heatmaps_p3 = output_p3['heatmaps']
    uncertainty = output_p3.get('uncertainty')
    
    print(f"   Output heatmaps: {heatmaps_p3.shape}")
    print(f"   Uncertainty: {uncertainty.shape if uncertainty is not None else 'N/A'}")
    assert heatmaps_p3.shape == (2, 20, 512, 512)
    assert uncertainty is not None and uncertainty.shape == (2, 20, 2)
    print(f"   Uncertainty range: [{uncertainty.min():.2f}, {uncertainty.max():.2f}]")
    print("   ✓ Phase 3 passed")
    
    # Test LA view
    print("\n4. Testing LA view (22 keypoints)...")
    model_la = create_maht_net(view='LA', phase=3)
    
    with torch.no_grad():
        output_la = model_la(x)
    
    print(f"   Output heatmaps: {output_la['heatmaps'].shape}")
    print(f"   Uncertainty: {output_la['uncertainty'].shape}")
    assert output_la['heatmaps'].shape == (2, 22, 512, 512)
    assert output_la['uncertainty'].shape == (2, 22, 2)
    print("   ✓ LA view passed")
    
    # Test keypoint extraction (soft-argmax)
    print("\n5. Testing keypoint extraction (soft-argmax)...")
    keypoints = model_p3.extract_keypoints(heatmaps_p3)
    print(f"   Extracted keypoints: {keypoints.shape}")
    assert keypoints.shape == (2, 20, 2)
    print("   ✓ Soft-argmax extraction passed")
    
    # Test DARK decoding
    print("\n6. Testing DARK decoding...")
    keypoints_dark, confidence = model_p3.extract_keypoints_dark(heatmaps_p3)
    print(f"   DARK keypoints: {keypoints_dark.shape}")
    print(f"   Confidence: {confidence.shape}")
    assert keypoints_dark.shape == (2, 20, 2)
    assert confidence.shape == (2, 20)
    print("   ✓ DARK decoding passed")
    
    # Test loss computation
    print("\n7. Testing loss computation...")
    gt_heatmaps = torch.randn_like(heatmaps_p3)
    loss = model_p3.compute_loss(heatmaps_p3, gt_heatmaps)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Loss computation passed")
    
    print("\n" + "#"*60)
    print("# All MAHT-Net tests passed! ✓")
    print("#"*60)
    return True


if __name__ == "__main__":
    test_maht_net()