# Component 4: Multi-Scale Decoder

## Overview

The Multi-Scale Decoder generates high-resolution heatmaps from the VAM-attended features. It uses **skip connections** from the CNN backbone to recover fine-grained spatial details lost during downsampling.

## Why Multi-Scale?

Keypoint detection requires:
1. **Semantic understanding** (which vertebra? which corner?) - from deep features
2. **Precise localization** (exact pixel location) - from shallow features

The decoder bridges this gap by progressively upsampling while incorporating multi-scale features.

## Architecture

```
VAM Output (K × 256)
        │
        ├──────────────── Reshape + Project ─────────────────┐
        ▼                                                     │
┌─────────────────────────────────────────────────────────────┘
│   Intermediate: K × 256 → 16×16×K (projected to spatial)
│
└───────────────────────────────────────────────────────────────┐
        │                                                       │
        ▼                                                       │
┌───────────────────────────────────────────────────────────────┘
│       DECODER STAGE 1: 16×16 → 32×32                         │
│       ┌──────────────────────────────────────────────────┐   │
│       │  Upsample 2× (bilinear)                          │   │
│       │  Concat with F3 (64×64 → crop/resize to 32×32)   │   │
│       │  Conv 3×3, BN, ReLU                              │   │
│       │  Conv 3×3, BN, ReLU                              │   │
│       └──────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│       DECODER STAGE 2: 32×32 → 64×64                         │
│       ┌──────────────────────────────────────────────────┐   │
│       │  Upsample 2×                                     │   │
│       │  Concat with F3 (64×64)                          │   │
│       │  Conv 3×3, BN, ReLU × 2                          │   │
│       └──────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│       DECODER STAGE 3: 64×64 → 128×128                       │
│       ┌──────────────────────────────────────────────────┐   │
│       │  Upsample 2×                                     │   │
│       │  Concat with F2 (128×128)                        │   │
│       │  Conv 3×3, BN, ReLU × 2                          │   │
│       └──────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│       DECODER STAGE 4: 128×128 → 256×256                     │
│       ┌──────────────────────────────────────────────────┐   │
│       │  Upsample 2×                                     │   │
│       │  Concat with F1 (256×256)                        │   │
│       │  Conv 3×3, BN, ReLU × 2                          │   │
│       └──────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│       FINAL STAGE: 256×256 → 512×512                         │
│       ┌──────────────────────────────────────────────────┐   │
│       │  Upsample 2×                                     │   │
│       │  Conv 3×3, BN, ReLU                              │   │
│       │  Conv 1×1 → K channels (one per keypoint)        │   │
│       └──────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Output Heatmaps (K × 512 × 512)
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    """
    Single decoder stage: Upsample → Concat skip → ConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        """
        Args:
            in_channels: Channels from previous decoder stage
            skip_channels: Channels from encoder skip connection
            out_channels: Output channels
            scale_factor: Upsampling factor
        """
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Convolution after concatenation
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features from previous stage [B, C, H, W]
            skip: Skip connection from encoder [B, C_skip, H_skip, W_skip]
            
        Returns:
            Upsampled and refined features
        """
        # Upsample
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        # Ensure spatial dimensions match (handle rounding issues)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        
        return x


class QueryToSpatial(nn.Module):
    """
    Projects VAM query outputs to spatial feature maps.
    
    Takes K × D query features and produces H × W × C spatial features
    that can be decoded into heatmaps.
    """
    
    def __init__(
        self,
        num_keypoints: int,
        query_dim: int,
        spatial_size: tuple,
        out_channels: int
    ):
        """
        Args:
            num_keypoints: Number of keypoints (K)
            query_dim: Query dimension (D)
            spatial_size: Target spatial size (H, W)
            out_channels: Output channels
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.spatial_size = spatial_size
        H, W = spatial_size
        
        # Project queries to spatial
        self.query_proj = nn.Linear(query_dim, H * W)
        
        # Channel projection
        self.channel_proj = nn.Conv2d(num_keypoints, out_channels, 1)
        
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: VAM output [B, K, D]
            
        Returns:
            Spatial features [B, C, H, W]
        """
        B, K, D = queries.shape
        H, W = self.spatial_size
        
        # Project to spatial [B, K, H*W]
        x = self.query_proj(queries)
        
        # Reshape to spatial [B, K, H, W]
        x = x.reshape(B, K, H, W)
        
        # Project channels [B, C, H, W]
        x = self.channel_proj(x)
        
        return x


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder with skip connections.
    
    Generates full-resolution heatmaps from VAM queries,
    using skip connections from the CNN backbone.
    """
    
    def __init__(
        self,
        num_keypoints: int,
        query_dim: int = 256,
        backbone_channels: list = [24, 48, 64, 256],
        decoder_channels: list = [256, 128, 64, 32],
        output_size: int = 512
    ):
        """
        Args:
            num_keypoints: Number of output heatmaps
            query_dim: VAM query dimension
            backbone_channels: Channels from CNN at each scale [F1, F2, F3, F4]
            decoder_channels: Decoder channels at each stage
            output_size: Target output resolution
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.output_size = output_size
        
        # Query to spatial projection
        self.query_to_spatial = QueryToSpatial(
            num_keypoints=num_keypoints,
            query_dim=query_dim,
            spatial_size=(16, 16),
            out_channels=decoder_channels[0]
        )
        
        # Decoder stages (progressively upsample)
        # 16→32, 32→64, 64→128, 128→256
        self.stages = nn.ModuleList([
            DecoderStage(decoder_channels[0], backbone_channels[2], decoder_channels[1]),  # +F3
            DecoderStage(decoder_channels[1], backbone_channels[2], decoder_channels[1]),  # +F3 again for 64
            DecoderStage(decoder_channels[1], backbone_channels[1], decoder_channels[2]),  # +F2
            DecoderStage(decoder_channels[2], backbone_channels[0], decoder_channels[3]),  # +F1
        ])
        
        # Final upsampling and output
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Heatmap head
        self.heatmap_head = nn.Conv2d(decoder_channels[3], num_keypoints, 1)
        
    def forward(
        self,
        queries: torch.Tensor,
        backbone_features: dict
    ) -> torch.Tensor:
        """
        Generate heatmaps from VAM queries and backbone features.
        
        Args:
            queries: VAM output [B, K, D]
            backbone_features: Dict with F1, F2, F3, F4 from CNN
            
        Returns:
            Heatmaps [B, K, H, W]
        """
        # Project queries to spatial
        x = self.query_to_spatial(queries)  # [B, 256, 16, 16]
        
        # Get skip connections
        f1 = backbone_features['F1']  # 256×256
        f2 = backbone_features['F2']  # 128×128
        f3 = backbone_features['F3']  # 64×64
        
        # Decoder path
        # Stage 1: 16→32 with F3 (resized)
        f3_small = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.stages[0](x, f3_small)  # [B, 128, 32, 32]
        
        # Stage 2: 32→64 with F3
        x = self.stages[1](x, f3)  # [B, 128, 64, 64]
        
        # Stage 3: 64→128 with F2
        x = self.stages[2](x, f2)  # [B, 64, 128, 128]
        
        # Stage 4: 128→256 with F1
        x = self.stages[3](x, f1)  # [B, 32, 256, 256]
        
        # Final: 256→512
        x = self.final_upsample(x)  # [B, 32, 512, 512]
        
        # Generate heatmaps
        heatmaps = self.heatmap_head(x)  # [B, K, 512, 512]
        
        return heatmaps


class SimpleDecoder(nn.Module):
    """
    Simpler decoder without query-to-spatial projection.
    
    Directly upsamples global features with skip connections.
    Used when VAM outputs are processed differently.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_keypoints: int = 20,
        backbone_channels: list = [24, 48, 64, 256]
    ):
        super().__init__()
        
        # Decoder stages
        self.up1 = DecoderStage(in_channels, backbone_channels[2], 128)    # 16→32
        self.up2 = DecoderStage(128, backbone_channels[2], 128)            # 32→64
        self.up3 = DecoderStage(128, backbone_channels[1], 64)             # 64→128
        self.up4 = DecoderStage(64, backbone_channels[0], 32)              # 128→256
        
        # Final output
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_keypoints, 1)
        )
        
    def forward(self, x: torch.Tensor, features: dict) -> torch.Tensor:
        """
        Args:
            x: Global features [B, C, 16, 16]
            features: Backbone features dict
        """
        f3_32 = F.interpolate(features['F3'], size=(32, 32), mode='bilinear', align_corners=False)
        
        x = self.up1(x, f3_32)
        x = self.up2(x, features['F3'])
        x = self.up3(x, features['F2'])
        x = self.up4(x, features['F1'])
        x = self.final(x)
        
        return x


# Usage example
if __name__ == "__main__":
    # Create decoder
    decoder = MultiScaleDecoder(
        num_keypoints=20,
        query_dim=256,
        output_size=512
    )
    
    # Simulate inputs
    batch_size = 4
    queries = torch.randn(batch_size, 20, 256)
    backbone_features = {
        'F1': torch.randn(batch_size, 24, 256, 256),
        'F2': torch.randn(batch_size, 48, 128, 128),
        'F3': torch.randn(batch_size, 64, 64, 64),
        'F4': torch.randn(batch_size, 256, 16, 16)
    }
    
    # Forward pass
    heatmaps = decoder(queries, backbone_features)
    
    print(f"Input queries: {queries.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")  # [4, 20, 512, 512]
    print(f"Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
```

## Heatmap Generation (Ground Truth)

For training, we generate ground truth heatmaps as 2D Gaussians centered on keypoint locations:

```python
def generate_heatmap(
    keypoints: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Generate Gaussian heatmaps for keypoints.
    
    Args:
        keypoints: [K, 2] tensor of (x, y) coordinates
        height: Heatmap height
        width: Heatmap width
        sigma: Gaussian standard deviation
        
    Returns:
        Heatmaps [K, H, W]
    """
    K = keypoints.shape[0]
    heatmaps = torch.zeros(K, height, width)
    
    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing='ij'
    )
    
    for k in range(K):
        x, y = keypoints[k]
        
        # Gaussian centered at (x, y)
        gaussian = torch.exp(
            -((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2)
        )
        
        heatmaps[k] = gaussian
    
    return heatmaps


def generate_heatmap_batch(
    keypoints_batch: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Batch version of heatmap generation.
    
    Args:
        keypoints_batch: [B, K, 2] batch of keypoints
        
    Returns:
        Heatmaps [B, K, H, W]
    """
    B, K, _ = keypoints_batch.shape
    heatmaps = torch.zeros(B, K, height, width, device=keypoints_batch.device)
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=keypoints_batch.device),
        torch.arange(width, device=keypoints_batch.device),
        indexing='ij'
    )
    y_grid = y_grid.float()
    x_grid = x_grid.float()
    
    for b in range(B):
        for k in range(K):
            x, y = keypoints_batch[b, k]
            
            gaussian = torch.exp(
                -((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2)
            )
            
            heatmaps[b, k] = gaussian
    
    return heatmaps
```

## Key Design Decisions

### 1. Skip Connection Strategy

**Decision**: Use all backbone features (F1, F2, F3).

**Rationale**:
- F1 (high-res): Edge details for precise localization
- F2 (mid-res): Local structure
- F3 (low-res): Semantic guidance

### 2. Upsampling Method

**Decision**: Bilinear interpolation (not transposed convolution).

**Rationale**:
- Avoids checkerboard artifacts
- More stable training
- Slightly better results in practice

### 3. Output Resolution

**Decision**: Full resolution (512×512).

**Rationale**:
- Sub-pixel accuracy via DARK decoding requires high-res heatmaps
- 128×128 output (common in pose estimation) is insufficient for mm-level accuracy

### 4. Gaussian Sigma

**Decision**: σ = 2 pixels.

**Rationale**:
- Small sigma → precise supervision, harder to learn
- Large sigma → easier to learn, less precise
- σ = 2 is a good balance for 512×512 output

## Memory Considerations

For batch size 8 and 512×512 output:

| Stage | Shape | Memory (fp32) |
|-------|-------|---------------|
| After query_to_spatial | 8×256×16×16 | 2 MB |
| After stage 1 | 8×128×32×32 | 4 MB |
| After stage 2 | 8×128×64×64 | 16 MB |
| After stage 3 | 8×64×128×128 | 32 MB |
| After stage 4 | 8×32×256×256 | 64 MB |
| Output heatmaps | 8×20×512×512 | 160 MB |
| **Total** | - | **~300 MB** |

This is manageable but significant. For Colab, batch size 4-8 is recommended.

## References

1. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Newell, A., et al. (2016). Stacked Hourglass Networks for Human Pose Estimation. ECCV.
3. Sun, K., et al. (2019). Deep High-Resolution Representation Learning for Visual Recognition. CVPR. (HRNet)
