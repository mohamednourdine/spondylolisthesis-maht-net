"""
HRNet-W32 backbone with heatmap head for vertebra keypoint detection.

Architecture:
    - HRNet-W32 encoder (pretrained on ImageNet via timm)
    - Multi-resolution parallel streams maintained throughout
    - Simple upsampling + heatmap head
    - Output: heatmaps at full resolution (512×512×40)

Reference:
    Sun, K., et al. "Deep High-Resolution Representation Learning for 
    Human Pose Estimation." CVPR 2019.

Key differences from ResNet:
    - HRNet maintains high-resolution features throughout (no severe downsampling)
    - Multi-scale fusion at every stage
    - Better for dense prediction tasks like keypoint localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Tuple


class HRNetHeatmap(nn.Module):
    """
    HRNet-W32 backbone with heatmap regression head for vertebra keypoint detection.
    
    This architecture leverages HRNet's multi-resolution parallel streams for
    maintaining spatial precision throughout the network. Unlike ResNet which
    aggressively downsamples and then upsamples, HRNet maintains high-resolution
    representations throughout the forward pass.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        num_keypoints: Number of output heatmap channels (40 = 10 vertebrae × 4 corners)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone_stages: Number of stages to freeze (0-4)
            0 = freeze nothing (train full model)
            1 = freeze stem + stage1
            2 = freeze stem + stage1-2
            etc.
        dropout_rate: Dropout rate for head (default: 0.3)
        output_size: Target output heatmap size (default: 512)
    
    Input: [B, 3, 512, 512]
    Output: [B, 40, 512, 512] (heatmaps, no activation)
    
    HRNet-W32 Feature Output Shapes (for 512x512 input):
        Stage 0: [64, 256, 256]   - 1/2 resolution
        Stage 1: [128, 128, 128]  - 1/4 resolution (we use this)
        Stage 2: [256, 64, 64]    - 1/8 resolution
        Stage 3: [512, 32, 32]    - 1/16 resolution
        Stage 4: [1024, 16, 16]   - 1/32 resolution (aggregated features)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 40,
        pretrained: bool = True,
        freeze_backbone_stages: int = 0,
        dropout_rate: float = 0.3,
        output_size: int = 512
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.pretrained = pretrained
        self.freeze_backbone_stages = freeze_backbone_stages
        self.output_size = output_size
        
        # Load HRNet-W32 backbone with timm
        # features_only=True returns intermediate features
        print("   Loading HRNet-W32 backbone...")
        self.backbone = timm.create_model(
            'hrnet_w32',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)  # Get all feature levels
        )
        
        if pretrained:
            print("   ✓ Loaded ImageNet pretrained weights")
        else:
            print("   ✓ Initialized from scratch (no pretrained weights)")
        
        # Feature info from timm:
        # Channels: [64, 128, 256, 512, 1024]
        # Reduction: [2, 4, 8, 16, 32]
        self.feature_channels = self.backbone.feature_info.channels()
        self.feature_reduction = self.backbone.feature_info.reduction()
        
        # Use highest resolution features (Stage 0: 64ch, 1/2 input)
        # OR use Stage 1 (128ch, 1/4 input) for better feature richness
        # We'll use Stage 1 (1/4 resolution = 128x128 for 512 input)
        self.use_stage = 1  # Index of stage to use for output
        feature_dim = self.feature_channels[self.use_stage]  # 128 channels
        
        # Heatmap head: refine features and predict heatmaps
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1)  # Final 1x1 conv
        )
        
        # Freeze backbone if specified
        if freeze_backbone_stages > 0:
            self._freeze_backbone(freeze_backbone_stages)
        
        # Initialize head weights
        self._init_head_weights()
        
        # Print model info
        self._print_model_info()
    
    def _freeze_backbone(self, num_stages: int):
        """Freeze backbone stages progressively."""
        frozen_params = 0
        
        # HRNet structure in timm: stem -> stages
        # We'll freeze by parameter count/name patterns
        for name, param in self.backbone.named_parameters():
            # Check if parameter should be frozen based on stage
            should_freeze = False
            
            # Freeze stem (conv1, bn1, conv2, bn2)
            if 'conv1' in name or 'bn1' in name or 'conv2' in name or 'bn2' in name:
                should_freeze = num_stages >= 1
            
            # Freeze stage1 (layer1)
            elif 'layer1' in name or 'stage1' in name:
                should_freeze = num_stages >= 1
            
            # Freeze stage2
            elif 'stage2' in name or 'transition1' in name:
                should_freeze = num_stages >= 2
            
            # Freeze stage3
            elif 'stage3' in name or 'transition2' in name:
                should_freeze = num_stages >= 3
            
            # Freeze stage4
            elif 'stage4' in name or 'transition3' in name:
                should_freeze = num_stages >= 4
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"   Frozen {frozen_params / 1e6:.2f}M backbone parameters (stages 0-{num_stages})")
    
    def _init_head_weights(self):
        """Initialize head weights using Kaiming initialization."""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final layer with small weights
        final_conv = self.head[-1]
        nn.init.normal_(final_conv.weight, std=0.001)
        nn.init.constant_(final_conv.bias, 0)
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"   Frozen parameters: {frozen_params / 1e6:.2f}M")
        print(f"   Backbone: {backbone_params / 1e6:.2f}M | Head: {head_params / 1e6:.2f}M")
        print(f"   Using Stage {self.use_stage} features ({self.feature_channels[self.use_stage]}ch, 1/{self.feature_reduction[self.use_stage]} resolution)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, 512, 512]
            
        Returns:
            Heatmaps [B, 40, 512, 512]
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Use selected stage features (Stage 1: 128ch, 1/4 resolution)
        feat = features[self.use_stage]  # [B, 128, 128, 128]
        
        # Generate heatmaps
        heatmaps = self.head(feat)  # [B, 40, 128, 128]
        
        # Upsample to target resolution
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 40, 512, 512]
        
        return heatmaps
    
    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> List[dict]:
        """
        Get parameter groups for differential learning rates.
        
        Args:
            backbone_lr: Learning rate for backbone (typically smaller)
            head_lr: Learning rate for head
            
        Returns:
            List of parameter group dicts for optimizer
        """
        # Backbone parameters (only trainable ones)
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        
        # Head parameters
        head_params = list(self.head.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': head_lr, 'name': 'head'},
        ]
        
        return param_groups
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation.
        
        Args:
            x: Input tensor [B, C, H, W]
            n_samples: Number of MC samples
            
        Returns:
            mean_heatmaps: Mean prediction [B, K, H, W]
            std_heatmaps: Standard deviation (uncertainty) [B, K, H, W]
            all_samples: All predictions [n_samples, B, K, H, W]
        """
        was_training = self.training
        self.train()  # Enable dropout for MC sampling
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                heatmaps = self.forward(x)
                samples.append(heatmaps)
        
        # Restore original training mode
        if not was_training:
            self.eval()
        
        # Stack samples: [n_samples, B, K, H, W]
        all_samples = torch.stack(samples, dim=0)
        
        # Calculate mean and std across samples
        mean_heatmaps = all_samples.mean(dim=0)
        std_heatmaps = all_samples.std(dim=0)
        
        return mean_heatmaps, std_heatmaps, all_samples


class HRNetHeatmapMultiScale(HRNetHeatmap):
    """
    HRNet-W32 with multi-scale feature fusion.
    
    This variant fuses features from multiple resolution streams before
    generating heatmaps. This can capture both fine details and global context.
    
    Uses features from Stage 0 (1/2 res), Stage 1 (1/4 res), and Stage 2 (1/8 res)
    and fuses them to the highest resolution.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 40,
        pretrained: bool = True,
        freeze_backbone_stages: int = 0,
        dropout_rate: float = 0.3,
        output_size: int = 512
    ):
        # Don't call parent __init__ yet - we'll set up differently
        nn.Module.__init__(self)
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.pretrained = pretrained
        self.freeze_backbone_stages = freeze_backbone_stages
        self.output_size = output_size
        
        # Load HRNet-W32 backbone
        print("   Loading HRNet-W32 backbone (Multi-Scale variant)...")
        self.backbone = timm.create_model(
            'hrnet_w32',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2)  # Use first 3 stages
        )
        
        if pretrained:
            print("   ✓ Loaded ImageNet pretrained weights")
        
        # Feature channels: [64, 128, 256]
        self.feature_channels = self.backbone.feature_info.channels()[:3]
        self.feature_reduction = self.backbone.feature_info.reduction()[:3]
        
        # Lateral convs to align channel dimensions
        fusion_dim = 128
        self.lateral_conv0 = nn.Conv2d(64, fusion_dim, kernel_size=1)
        self.lateral_conv1 = nn.Conv2d(128, fusion_dim, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(256, fusion_dim, kernel_size=1)
        
        # Fusion conv after concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim * 3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )
        
        # Heatmap head
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1)
        )
        
        # Freeze backbone if specified
        if freeze_backbone_stages > 0:
            self._freeze_backbone(freeze_backbone_stages)
        
        # Initialize weights
        self._init_weights()
        
        # Print info
        self._print_model_info()
    
    def _init_weights(self):
        """Initialize fusion and head weights."""
        for m in [self.lateral_conv0, self.lateral_conv1, self.lateral_conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        
        for m in self.fusion_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Final layer with small weights
        final_conv = self.head[-1]
        nn.init.normal_(final_conv.weight, std=0.001)
        nn.init.constant_(final_conv.bias, 0)
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        
        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"   Backbone: {backbone_params / 1e6:.2f}M")
        print(f"   Multi-scale fusion: Stage 0-2 features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale feature fusion.
        
        Args:
            x: Input tensor [B, 3, 512, 512]
            
        Returns:
            Heatmaps [B, 40, 512, 512]
        """
        # Extract multi-scale features
        features = self.backbone(x)
        # features[0]: [B, 64, 256, 256]   - 1/2 resolution
        # features[1]: [B, 128, 128, 128]  - 1/4 resolution
        # features[2]: [B, 256, 64, 64]    - 1/8 resolution
        
        # Apply lateral convs
        f0 = self.lateral_conv0(features[0])  # [B, 128, 256, 256]
        f1 = self.lateral_conv1(features[1])  # [B, 128, 128, 128]
        f2 = self.lateral_conv2(features[2])  # [B, 128, 64, 64]
        
        # Upsample all to highest resolution (1/2)
        target_size = f0.shape[2:]  # (256, 256)
        f1_up = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2_up = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        fused = torch.cat([f0, f1_up, f2_up], dim=1)  # [B, 384, 256, 256]
        fused = self.fusion_conv(fused)  # [B, 256, 256, 256]
        
        # Generate heatmaps
        heatmaps = self.head(fused)  # [B, 40, 256, 256]
        
        # Upsample to target resolution
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 40, 512, 512]
        
        return heatmaps
    
    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> List[dict]:
        """Get parameter groups for differential learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        
        fusion_params = (
            list(self.lateral_conv0.parameters()) +
            list(self.lateral_conv1.parameters()) +
            list(self.lateral_conv2.parameters()) +
            list(self.fusion_conv.parameters())
        )
        
        head_params = list(self.head.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': fusion_params, 'lr': head_lr, 'name': 'fusion'},
            {'params': head_params, 'lr': head_lr, 'name': 'head'},
        ]
        
        return param_groups


def create_hrnet_heatmap(multi_scale: bool = False, **kwargs) -> nn.Module:
    """
    Factory function to create HRNet heatmap model.
    
    Args:
        multi_scale: If True, use multi-scale fusion variant
        in_channels: Number of input channels (default: 3)
        num_keypoints: Number of output channels (default: 40)
        pretrained: Use ImageNet weights (default: True)
        freeze_backbone_stages: Stages to freeze (default: 0)
        dropout_rate: Head dropout (default: 0.3)
        output_size: Output heatmap size (default: 512)
        
    Returns:
        HRNetHeatmap or HRNetHeatmapMultiScale model instance
    """
    if multi_scale:
        return HRNetHeatmapMultiScale(**kwargs)
    return HRNetHeatmap(**kwargs)


# For testing
if __name__ == '__main__':
    print("=" * 60)
    print("Testing HRNetHeatmap model...")
    print("=" * 60)
    
    # Create model (standard version)
    print("\n1. Standard HRNetHeatmap (single-scale):")
    model = HRNetHeatmap(
        in_channels=3,
        num_keypoints=40,
        pretrained=False,  # Don't download weights for test
        freeze_backbone_stages=0,
        dropout_rate=0.3
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    print(f"\n   Input shape: {x.shape}")
    
    y = model(x)
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # Test parameter groups
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
    print("\n   Parameter groups:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"     {pg['name']}: {n_params/1e6:.2f}M params, lr={pg['lr']}")
    
    # Test multi-scale version
    print("\n" + "=" * 60)
    print("2. Multi-Scale HRNetHeatmap:")
    print("=" * 60)
    
    model_ms = HRNetHeatmapMultiScale(
        in_channels=3,
        num_keypoints=40,
        pretrained=False,
        freeze_backbone_stages=0,
        dropout_rate=0.3
    )
    
    y_ms = model_ms(x)
    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {y_ms.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
