# Component 1: CNN Backbone (EfficientNetV2-S)

## Overview

The CNN backbone extracts hierarchical visual features from X-ray images. We use **EfficientNetV2-S** pretrained on ImageNet, which provides an excellent balance between accuracy and computational efficiency.

## Why EfficientNetV2?

| Criterion | EfficientNetV2-S | ResNet-50 | ConvNeXt-S |
|-----------|------------------|-----------|------------|
| **ImageNet Top-1** | 83.9% | 76.1% | 83.1% |
| **Parameters** | 21M | 25M | 50M |
| **FLOPs** | 8.4B | 4.1B | 8.7B |
| **Training Speed** | Fast | Medium | Slow |
| **Colab Friendly** | ✅ Yes | ✅ Yes | ⚠️ Marginal |

**Decision**: EfficientNetV2-S offers state-of-the-art accuracy with reasonable compute, making it ideal for our Colab training environment.

## Architecture Details

### EfficientNetV2-S Structure

```
Input: 512×512×3
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 0: Stem                                              │
│  Conv 3×3, stride 2 → BN → SiLU                            │
│  Output: 256×256×24                                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Fused-MBConv ×2                                   │
│  Output: 256×256×24  → F1 (skip connection)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Fused-MBConv ×4                                   │
│  Output: 128×128×48  → F2 (skip connection)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Fused-MBConv ×4                                   │
│  Output: 64×64×64    → F3 (skip connection)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: MBConv ×6                                         │
│  Output: 32×32×128                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: MBConv ×9                                         │
│  Output: 16×16×160                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: MBConv ×15                                        │
│  Output: 16×16×256   → F4 (to Transformer)                 │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Scale Feature Outputs

We extract features at 4 scales for the decoder skip connections:

| Feature | Resolution | Channels | Scale | Use |
|---------|------------|----------|-------|-----|
| **F1** | 256×256 | 24 | 1/2 | Fine details (edges) |
| **F2** | 128×128 | 48 | 1/4 | Low-level structure |
| **F3** | 64×64 | 64 | 1/8 | Mid-level features |
| **F4** | 16×16 | 256 | 1/32 | High-level semantics |

## Implementation

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2-S backbone with multi-scale feature extraction.
    
    Extracts features at 4 scales for skip connections to decoder.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: int = 2,
        out_channels: int = 256
    ):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            freeze_stages: Number of stages to freeze (0-6)
            out_channels: Output channels after projection (for Transformer)
        """
        super().__init__()
        
        # Load pretrained EfficientNetV2-S
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_v2_s(weights=weights)
        else:
            self.backbone = efficientnet_v2_s(weights=None)
        
        # Remove classifier head
        self.backbone.classifier = nn.Identity()
        
        # Feature extraction hooks
        self.features = {}
        self._register_hooks()
        
        # Freeze early stages
        self._freeze_stages(freeze_stages)
        
        # Project F4 to desired channels for Transformer
        self.proj = nn.Conv2d(256, out_channels, kernel_size=1)
        
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Hook into specific stages
        # Stage 1 output (after block 1)
        self.backbone.features[1].register_forward_hook(get_hook('F1'))
        # Stage 2 output (after block 2)
        self.backbone.features[2].register_forward_hook(get_hook('F2'))
        # Stage 3 output (after block 3)
        self.backbone.features[3].register_forward_hook(get_hook('F3'))
        # Stage 6 output (final features)
        self.backbone.features[6].register_forward_hook(get_hook('F4'))
    
    def _freeze_stages(self, num_stages: int):
        """Freeze the first num_stages stages."""
        if num_stages <= 0:
            return
            
        # Freeze stem
        for param in self.backbone.features[0].parameters():
            param.requires_grad = False
            
        # Freeze specified stages
        for i in range(1, min(num_stages + 1, 7)):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Extract multi-scale features.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary with F1, F2, F3, F4 features
        """
        # Clear previous features
        self.features = {}
        
        # Forward pass (hooks capture features)
        _ = self.backbone(x)
        
        # Project F4 for Transformer input
        self.features['F4'] = self.proj(self.features['F4'])
        
        return self.features


# Alternative: Manual feature extraction (more explicit)
class EfficientNetV2BackboneExplicit(nn.Module):
    """
    Alternative implementation with explicit feature extraction.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            backbone = efficientnet_v2_s(weights=weights)
        else:
            backbone = efficientnet_v2_s(weights=None)
        
        # Split into stages
        self.stem = backbone.features[0]      # Conv stem
        self.stage1 = backbone.features[1]    # Fused-MBConv
        self.stage2 = backbone.features[2]    # Fused-MBConv
        self.stage3 = backbone.features[3]    # Fused-MBConv
        self.stage4 = backbone.features[4]    # MBConv
        self.stage5 = backbone.features[5]    # MBConv
        self.stage6 = backbone.features[6]    # MBConv
        
        # Channel projections for uniform decoder input
        self.proj1 = nn.Conv2d(24, 64, 1)
        self.proj2 = nn.Conv2d(48, 128, 1)
        self.proj3 = nn.Conv2d(64, 256, 1)
        self.proj4 = nn.Conv2d(256, 512, 1)
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stage 1
        x = self.stage1(x)
        f1 = self.proj1(x)  # 256×256×64
        
        # Stage 2
        x = self.stage2(x)
        f2 = self.proj2(x)  # 128×128×128
        
        # Stage 3
        x = self.stage3(x)
        f3 = self.proj3(x)  # 64×64×256
        
        # Stages 4-6
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        f4 = self.proj4(x)  # 16×16×512
        
        return {'F1': f1, 'F2': f2, 'F3': f3, 'F4': f4}
```

## Key Design Decisions

### 1. Pretrained Weights

**Decision**: Use ImageNet pretrained weights.

**Rationale**: 
- Medical imaging benefits from transfer learning despite domain gap
- Low-level features (edges, textures) transfer well
- Faster convergence and better generalization with limited data (3,600 patients)

### 2. Stage Freezing

**Decision**: Freeze first 2 stages during initial training.

**Rationale**:
- Early layers learn general features (edges, textures) that transfer well
- Reduces risk of overfitting on small medical dataset
- Allows later layers to adapt to X-ray specific patterns
- Unfrozen in later training phases for fine-tuning

### 3. Multi-Scale Features

**Decision**: Extract features at 4 resolutions.

**Rationale**:
- Keypoint detection requires both fine details (exact corner location) and global context (vertebra identity)
- Skip connections enable precise localization
- Different scales capture different anatomical structures

## Channel Dimensions Summary

| Stage | Input Channels | Output Channels | Resolution (512 input) |
|-------|----------------|-----------------|------------------------|
| Stem | 3 | 24 | 256×256 |
| Stage 1 | 24 | 24 | 256×256 |
| Stage 2 | 24 | 48 | 128×128 |
| Stage 3 | 48 | 64 | 64×64 |
| Stage 4 | 64 | 128 | 32×32 |
| Stage 5 | 128 | 160 | 16×16 |
| Stage 6 | 160 | 256 | 16×16 |

## Memory Footprint

For a batch size of 8 with 512×512 input:

| Feature | Shape | Memory (fp32) |
|---------|-------|---------------|
| F1 | 8×24×256×256 | 48 MB |
| F2 | 8×48×128×128 | 24 MB |
| F3 | 8×64×64×64 | 8 MB |
| F4 | 8×256×16×16 | 2 MB |
| **Total** | - | **~82 MB** |

This is well within Colab's 15GB GPU memory limit.

## References

1. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
