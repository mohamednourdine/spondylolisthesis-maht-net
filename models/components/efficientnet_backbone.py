"""
Component 1: EfficientNetV2-S Backbone
Extracts multi-scale features for MAHT-Net.

This module provides the CNN backbone for feature extraction,
returning multi-scale features at different resolutions for
skip connections in the decoder.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import List, Tuple, Dict


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2-S backbone with multi-scale feature extraction.
    
    Extracts features at 4 scales for decoder skip connections:
        F1: 1/4 resolution (128×128 for 512×512 input)
        F2: 1/8 resolution (64×64)
        F3: 1/16 resolution (32×32)
        F4: 1/32 resolution (16×16) → main output for transformer
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_stages: Number of early stages to freeze (0-4)
        out_channels: Output channels for F4 projection
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: int = 2,
        out_channels: int = 256
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Load pretrained model
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.model = efficientnet_v2_s(weights=weights)
            print("  ✓ Loaded EfficientNetV2-S with ImageNet weights")
        else:
            self.model = efficientnet_v2_s(weights=None)
            print("  ✓ Loaded EfficientNetV2-S without pretrained weights")
        
        # Extract feature stages from EfficientNetV2-S
        self.features = self.model.features
        
        # EfficientNetV2-S channel dimensions at each stage output
        # Stage 0: Conv2d(3, 24) + fused_mbconv -> 24 channels
        # Stage 1: fused_mbconv -> 24 channels, 1/2 resolution
        # Stage 2: fused_mbconv -> 48 channels, 1/4 resolution
        # Stage 3: fused_mbconv -> 64 channels, 1/8 resolution
        # Stage 4: mbconv -> 128 channels, 1/16 resolution
        # Stage 5: mbconv -> 160 channels, 1/16 resolution
        # Stage 6: mbconv -> 256 channels, 1/32 resolution
        # Stage 7: Conv2d -> 1280 channels, 1/32 resolution
        
        # We'll extract features at these stages:
        # F1: After stage 2 (1/4 res, 48 ch)
        # F2: After stage 3 (1/8 res, 64 ch)
        # F3: After stage 4 (1/16 res, 128 ch)
        # F4: After stage 6 (1/32 res, 256 ch)
        
        self.feature_channels = {
            'F1': 48,   # stage 2 output
            'F2': 64,   # stage 3 output
            'F3': 128,  # stage 4 output
            'F4': 256   # stage 6 output
        }
        
        # Projection layer for F4 if out_channels differs
        if out_channels != self.feature_channels['F4']:
            self.out_proj = nn.Conv2d(
                self.feature_channels['F4'], out_channels, 
                kernel_size=1, bias=False
            )
        else:
            self.out_proj = nn.Identity()
        
        # Freeze early stages if specified
        self._freeze_stages(freeze_stages)
    
    def _freeze_stages(self, num_stages: int):
        """Freeze early stages to preserve pretrained features."""
        if num_stages <= 0:
            return
        
        # Freeze the specified number of stages
        for i in range(min(num_stages + 1, len(self.features))):
            for param in self.features[i].parameters():
                param.requires_grad = False
        
        print(f"  ✓ Froze first {num_stages} stages of backbone")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multi-scale feature extraction.
        
        Args:
            x: Input tensor (B, 3, H, W), typically (B, 3, 512, 512)
            
        Returns:
            f4: High-level features (B, out_channels, H/32, W/32)
            skip_features: [F1, F2, F3] for decoder skip connections
        """
        skip_features = []
        
        # Process through feature stages
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Extract skip features at specific stages
            if i == 2:  # F1: 1/4 resolution
                skip_features.append(x)
            elif i == 3:  # F2: 1/8 resolution
                skip_features.append(x)
            elif i == 4:  # F3: 1/16 resolution
                skip_features.append(x)
            elif i == 6:  # F4: 1/32 resolution
                f4 = x
        
        # Project F4 to output channels
        f4 = self.out_proj(f4)
        
        return f4, skip_features
    
    def get_feature_channels(self) -> Dict[str, int]:
        """Return the channel dimensions for each feature level."""
        return self.feature_channels.copy()


def test_backbone():
    """Test the backbone module."""
    print("\nTesting EfficientNetV2 Backbone...")
    print("-" * 50)
    
    model = EfficientNetV2Backbone(pretrained=True, freeze_stages=2)
    
    # Test with typical input size
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        f4, skips = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"F4 output: {f4.shape}")
    print(f"\nSkip features:")
    for i, skip in enumerate(skips):
        print(f"  F{i+1}: {skip.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    print("\n✓ Backbone test passed!")
    return True


if __name__ == "__main__":
    test_backbone()
