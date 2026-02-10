# Step 2: MAHT-Net Model Implementation

## Overview

This guide explains how to implement the full MAHT-Net architecture. The current `models/maht_net.py` is a **stub** (~35 lines, basic CNN). We need to implement the complete architecture as documented in `docs/architecture/`.

**Estimated Time**: 5-7 days

---

## Architecture Summary

```
Input Image (512×512×3)
        │
        ▼
┌────────────────────────┐
│  1. CNN Backbone       │  EfficientNetV2-S (pretrained)
│     (EfficientNetV2)   │  → Multi-scale features F1, F2, F3, F4
└────────────────────────┘
        │
        ▼ F4 (16×16×256)
┌────────────────────────┐
│  2. Transformer Bridge │  4-layer encoder
│     (Global Context)   │  → Global spatial relationships
└────────────────────────┘
        │
        ▼ (16×16×256)
┌────────────────────────┐
│  3. Vertebral Attention│  K learnable queries (K=20 or 22)
│     Module (VAM)       │  → Anatomically-aware features
└────────────────────────┘
        │
        ▼ (K×256)
┌────────────────────────┐
│  4. Multi-scale        │  Skip connections from F1, F2, F3
│     Decoder            │  → High-resolution heatmaps
└────────────────────────┘
        │
        ▼ (K×512×512)
┌────────────────────────┐
│  5. DARK Decoding      │  Sub-pixel coordinate extraction
│     (Inference only)   │  → K keypoints (x, y, σ)
└────────────────────────┘
        │
        ▼
Output: K keypoints with confidence
```

---

## Step 2.1: File Structure

Create/update these files:

```
models/
├── maht_net.py           # Main model (REWRITE)
├── components/
│   ├── __init__.py
│   ├── efficientnet_backbone.py  # CNN backbone
│   ├── transformer_bridge.py     # Transformer encoder
│   ├── vam.py                    # Vertebral Attention Module
│   ├── decoder.py                # Multi-scale decoder
│   └── dark_decoding.py          # Coordinate extraction
```

Create the components folder:
```bash
mkdir -p models/components
touch models/components/__init__.py
```

---

## Step 2.2: CNN Backbone Implementation

Create `models/components/efficientnet_backbone.py`:

```python
"""
Component 1: EfficientNetV2-S Backbone
Extracts multi-scale features for MAHT-Net.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import List, Tuple


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2-S backbone with multi-scale feature extraction.
    
    Features extracted at 4 scales for decoder skip connections:
        F1: 256×256×24  (1/2 resolution)
        F2: 128×128×48  (1/4 resolution)
        F3: 64×64×64    (1/8 resolution)
        F4: 16×16×256   (1/32 resolution) → goes to Transformer
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_stages: Number of early stages to freeze (0-4)
        out_channels: Output channels for F4
    """
    
    # Feature extraction points (EfficientNetV2 stage indices)
    FEATURE_STAGES = {
        'F1': 1,   # After stage 1 (256×256)
        'F2': 2,   # After stage 2 (128×128)
        'F3': 3,   # After stage 3 (64×64)
        'F4': 6,   # After stage 6 (16×16)
    }
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: int = 2,
        out_channels: int = 256
    ):
        super().__init__()
        
        # Load pretrained model
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.model = efficientnet_v2_s(weights=weights)
            print("  ✓ Loaded EfficientNetV2-S with ImageNet weights")
        else:
            self.model = efficientnet_v2_s(weights=None)
            print("  ✓ Loaded EfficientNetV2-S without pretrained weights")
        
        # Get feature stages
        self.features = self.model.features
        
        # Channel dimensions at each stage
        self.feature_channels = {
            'F1': 24,
            'F2': 48,
            'F3': 64,
            'F4': 256
        }
        
        # Projection to uniform output channels
        self.out_proj = nn.Conv2d(self.feature_channels['F4'], out_channels, 1)
        
        # Freeze early stages if specified
        self._freeze_stages(freeze_stages)
    
    def _freeze_stages(self, num_stages: int):
        """Freeze early stages to preserve pretrained features."""
        if num_stages <= 0:
            return
        
        for i in range(min(num_stages, len(self.features))):
            for param in self.features[i].parameters():
                param.requires_grad = False
        
        print(f"  ✓ Froze first {num_stages} stages")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multi-scale feature extraction.
        
        Args:
            x: Input tensor (B, 3, 512, 512)
            
        Returns:
            f4: High-level features (B, out_channels, 16, 16)
            skip_features: [F1, F2, F3] for decoder skip connections
        """
        skip_features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Extract skip features
            if i == self.FEATURE_STAGES['F1']:
                skip_features.append(x)  # F1: 256×256×24
            elif i == self.FEATURE_STAGES['F2']:
                skip_features.append(x)  # F2: 128×128×48
            elif i == self.FEATURE_STAGES['F3']:
                skip_features.append(x)  # F3: 64×64×64
        
        # Project F4 to output channels
        f4 = self.out_proj(x)  # (B, out_channels, 16, 16)
        
        return f4, skip_features


def test_backbone():
    """Test the backbone module."""
    model = EfficientNetV2Backbone(pretrained=True)
    x = torch.randn(2, 3, 512, 512)
    
    f4, skips = model(x)
    
    print(f"Input: {x.shape}")
    print(f"F4: {f4.shape}")
    for i, skip in enumerate(skips):
        print(f"Skip {i+1}: {skip.shape}")


if __name__ == "__main__":
    test_backbone()
```

---

## Step 2.3: Transformer Bridge Implementation

Create `models/components/transformer_bridge.py`:

```python
"""
Component 2: Transformer Bridge
Models global spatial relationships in feature maps.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for spatial features."""
    
    def __init__(self, d_model: int, height: int = 16, width: int = 16):
        super().__init__()
        
        # Learnable position embeddings
        self.row_embed = nn.Parameter(torch.randn(height, d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(width, d_model // 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D positional encoding.
        
        Args:
            x: (B, H*W, D) flattened feature tokens
            
        Returns:
            x + position encoding
        """
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        
        # Create 2D position encoding
        pos_h = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)  # (H, W, D//2)
        pos_w = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)  # (H, W, D//2)
        
        pos = torch.cat([pos_h, pos_w], dim=-1)  # (H, W, D)
        pos = pos.view(1, N, D).expand(B, -1, -1)  # (B, N, D)
        
        return x + pos.to(x.device)


class TransformerBridge(nn.Module):
    """
    Transformer encoder for global context modeling.
    
    Processes flattened CNN features (16×16 → 256 tokens) through
    multi-head self-attention layers.
    
    Args:
        d_model: Transformer dimension (should match CNN output)
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_size: int = 16
    ):
        super().__init__()
        
        self.d_model = d_model
        self.feature_size = feature_size
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, feature_size, feature_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        print(f"  ✓ TransformerBridge: {num_layers} layers, {nhead} heads")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: CNN features (B, C, H, W) where C=d_model
            
        Returns:
            Global features (B, C, H, W) same shape as input
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, C) where N = H*W
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.encoder(x)
        x = self.norm(x)
        
        # Reshape back to spatial: (B, N, C) → (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        return x


def test_transformer():
    """Test the transformer bridge."""
    model = TransformerBridge(d_model=256, num_layers=4)
    x = torch.randn(2, 256, 16, 16)
    
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")


if __name__ == "__main__":
    test_transformer()
```

---

## Step 2.4: Vertebral Attention Module (VAM)

Create `models/components/vam.py`:

```python
"""
Component 3: Vertebral Attention Module (VAM)
Our key architectural contribution - anatomically-aware keypoint attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AnatomicalPositionEncoding(nn.Module):
    """
    Encodes anatomical priors about vertebral structure.
    
    Each keypoint is encoded with:
    - Vertebra level (L1, L2, L3, L4, L5, S1)
    - Corner type (left, right)
    - Edge type (upper, lower)
    """
    
    def __init__(self, d_model: int, num_vertebrae: int = 6, num_corners: int = 2):
        super().__init__()
        
        # Learnable embeddings for anatomical structure
        self.vertebra_embed = nn.Embedding(num_vertebrae, d_model // 4)
        self.edge_embed = nn.Embedding(2, d_model // 4)  # upper/lower
        self.corner_embed = nn.Embedding(num_corners, d_model // 4)  # left/right
        
        # Continuous position embedding (normalized y-position)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
    
    def forward(self, vertebra_ids: torch.Tensor, edge_ids: torch.Tensor,
                corner_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate anatomical position encoding.
        
        Args:
            vertebra_ids: (K,) vertebra indices (0=L1, 1=L2, ..., 5=S1)
            edge_ids: (K,) edge indices (0=upper, 1=lower)
            corner_ids: (K,) corner indices (0=left, 1=right)
            positions: (K, 1) normalized y-positions [0, 1]
            
        Returns:
            pos_encoding: (K, d_model)
        """
        v_embed = self.vertebra_embed(vertebra_ids)   # (K, d//4)
        e_embed = self.edge_embed(edge_ids)           # (K, d//4)
        c_embed = self.corner_embed(corner_ids)       # (K, d//4)
        p_embed = self.position_mlp(positions)        # (K, d//4)
        
        return torch.cat([v_embed, e_embed, c_embed, p_embed], dim=-1)


class VAM(nn.Module):
    """
    Vertebral Attention Module.
    
    Uses learnable keypoint queries with anatomical priors to attend
    to relevant positions in the global feature map.
    
    Args:
        d_model: Feature dimension
        num_keypoints: Number of keypoints (20 for AP, 22 for LA)
        nhead: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_keypoints: int = 20,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_keypoints = num_keypoints
        self.num_layers = num_layers
        
        # Learnable keypoint queries
        self.keypoint_queries = nn.Parameter(
            torch.randn(num_keypoints, d_model) * 0.02
        )
        
        # Anatomical position encoding
        self.anatomical_encoding = AnatomicalPositionEncoding(d_model)
        
        # Register anatomical indices as buffers
        self._register_anatomical_indices(num_keypoints)
        
        # Cross-attention layers (query attends to image features)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Self-attention layers (queries attend to each other)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        print(f"  ✓ VAM: {num_keypoints} keypoints, {num_layers} layers")
    
    def _register_anatomical_indices(self, num_keypoints: int):
        """Create anatomical index tensors for position encoding."""
        # For BUU-LSPINE: each row has left corner + right corner
        # Row 0: L1-upper-left, L1-upper-right
        # Row 1: L1-lower-left, L1-lower-right
        # etc.
        
        num_rows = num_keypoints // 2
        
        vertebra_ids = []
        edge_ids = []
        corner_ids = []
        positions = []
        
        for row in range(num_rows):
            vertebra = row // 2  # 0-4 for L1-L5, 5 for S1
            edge = row % 2       # 0=upper, 1=lower
            
            # Left corner
            vertebra_ids.append(vertebra)
            edge_ids.append(edge)
            corner_ids.append(0)  # left
            positions.append(row / (num_rows - 1))  # normalized position
            
            # Right corner
            vertebra_ids.append(vertebra)
            edge_ids.append(edge)
            corner_ids.append(1)  # right
            positions.append(row / (num_rows - 1))
        
        self.register_buffer('vertebra_ids', torch.tensor(vertebra_ids))
        self.register_buffer('edge_ids', torch.tensor(edge_ids))
        self.register_buffer('corner_ids', torch.tensor(corner_ids))
        self.register_buffer('positions', torch.tensor(positions).unsqueeze(-1))
    
    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAM.
        
        Args:
            global_features: (B, C, H, W) from Transformer Bridge
            
        Returns:
            keypoint_features: (B, K, C) attended features per keypoint
        """
        B, C, H, W = global_features.shape
        
        # Flatten spatial dimensions for attention
        memory = global_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Get keypoint queries with anatomical encoding
        anatomical_pos = self.anatomical_encoding(
            self.vertebra_ids, self.edge_ids, 
            self.corner_ids, self.positions
        )  # (K, C)
        
        queries = self.keypoint_queries + anatomical_pos  # (K, C)
        queries = queries.unsqueeze(0).expand(B, -1, -1)   # (B, K, C)
        
        # Apply attention layers
        for i in range(self.num_layers):
            # Cross-attention: queries attend to image features
            q = self.norm1[i](queries)
            queries = queries + self.cross_attn_layers[i](q, memory, memory)[0]
            
            # Self-attention: queries attend to each other
            q = self.norm2[i](queries)
            queries = queries + self.self_attn_layers[i](q, q, q)[0]
            
            # FFN
            queries = queries + self.ffn_layers[i](self.norm3[i](queries))
        
        return queries


def test_vam():
    """Test the VAM module."""
    model = VAM(d_model=256, num_keypoints=20, num_layers=3)
    x = torch.randn(2, 256, 16, 16)
    
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")  # Should be (2, 20, 256)


if __name__ == "__main__":
    test_vam()
```

---

## Step 2.5: Multi-scale Decoder

Create `models/components/decoder.py`:

```python
"""
Component 4: Multi-scale Decoder
Generates high-resolution heatmaps with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Upsample (2x)
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=4, stride=2, padding=1
        )
        
        # Skip connection projection
        self.skip_proj = nn.Conv2d(skip_channels, in_channels // 2, kernel_size=1)
        
        # Fusion convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature from previous decoder level
            skip: Skip connection from encoder
        """
        x = self.upsample(x)
        skip = self.skip_proj(skip)
        
        # Resize if needed (for dimension mismatch)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder for heatmap generation.
    
    Takes VAM output and skip features, produces K heatmaps at 512×512.
    
    Architecture:
        VAM output (B, K, C) → reshape → (B, C, 1, 1)
        Upsample through decoder blocks with skip connections
        → (B, K, 512, 512) output heatmaps
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        d_model: int = 256,
        skip_channels: List[int] = [24, 48, 64],  # F1, F2, F3 channels
        decoder_channels: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        
        # Project VAM output to spatial feature
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, decoder_channels * 16 * 16),
            nn.ReLU()
        )
        
        # Decoder blocks (progressively upsample)
        # 16→32: use global features
        # 32→64: skip from F3
        # 64→128: skip from F2
        # 128→256: skip from F1
        # 256→512: final upsample
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # 16→32 (no skip yet)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # 32→64 with F3 skip (64 channels)
        self.block2 = DecoderBlock(decoder_channels, skip_channels[2], decoder_channels, dropout)
        
        # 64→128 with F2 skip (48 channels)
        self.block3 = DecoderBlock(decoder_channels, skip_channels[1], decoder_channels // 2, dropout)
        
        # 128→256 with F1 skip (24 channels)
        self.block4 = DecoderBlock(decoder_channels // 2, skip_channels[0], decoder_channels // 4, dropout)
        
        # 256→512 final upsample
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Output head: produce K heatmaps
        self.output_head = nn.Conv2d(decoder_channels // 4, num_keypoints, 1)
        
        print(f"  ✓ Decoder: {num_keypoints} output channels")
    
    def forward(
        self,
        vam_output: torch.Tensor,
        skip_features: List[torch.Tensor],
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate heatmaps from VAM output and skip features.
        
        Args:
            vam_output: (B, K, C) from VAM
            skip_features: [F1, F2, F3] from backbone
            global_features: (B, C, H, W) from transformer
            
        Returns:
            heatmaps: (B, K, 512, 512)
        """
        B, K, C = vam_output.shape
        F1, F2, F3 = skip_features
        
        # Sum VAM outputs to create spatial seed (simplified approach)
        # More sophisticated: use each query to modulate corresponding heatmap
        x = vam_output.mean(dim=1)  # (B, C)
        
        # Project to spatial
        x = self.input_proj(x)  # (B, C*16*16)
        x = x.view(B, -1, 16, 16)  # (B, decoder_channels, 16, 16)
        
        # Combine with global features
        x = x + F.adaptive_avg_pool2d(global_features, (16, 16))
        x = self.initial_conv(x)
        
        # Decode with skip connections
        x = self.up1(x)          # 16→32
        x = self.block2(x, F3)   # 32→64
        x = self.block3(x, F2)   # 64→128
        x = self.block4(x, F1)   # 128→256
        x = self.final_up(x)     # 256→512
        
        # Output heatmaps
        heatmaps = self.output_head(x)  # (B, K, 512, 512)
        
        return heatmaps


def test_decoder():
    """Test the decoder."""
    model = MultiScaleDecoder(num_keypoints=20)
    
    vam_out = torch.randn(2, 20, 256)
    f1 = torch.randn(2, 24, 256, 256)
    f2 = torch.randn(2, 48, 128, 128)
    f3 = torch.randn(2, 64, 64, 64)
    global_feat = torch.randn(2, 256, 16, 16)
    
    out = model(vam_out, [f1, f2, f3], global_feat)
    print(f"Output: {out.shape}")  # Should be (2, 20, 512, 512)


if __name__ == "__main__":
    test_decoder()
```

---

## Step 2.6: Main MAHT-Net Model

Replace `models/maht_net.py` with:

```python
"""
MAHT-Net: Multi-scale Anatomical Heatmap Transformer

Full implementation of MAHT-Net for vertebral corner point detection.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .components.efficientnet_backbone import EfficientNetV2Backbone
from .components.transformer_bridge import TransformerBridge
from .components.vam import VAM
from .components.decoder import MultiScaleDecoder


class MAHTNet(nn.Module):
    """
    MAHT-Net: Multi-scale Anatomical Heatmap Transformer
    
    A deep learning architecture for precise vertebral corner point
    detection in lumbar spine X-ray images.
    
    Architecture:
        1. CNN Backbone (EfficientNetV2-S) - Multi-scale feature extraction
        2. Transformer Bridge - Global context modeling
        3. Vertebral Attention Module - Anatomically-aware queries
        4. Multi-scale Decoder - High-resolution heatmap generation
    
    Args:
        num_keypoints: Number of keypoints (20 for AP, 22 for LA)
        d_model: Feature dimension (256)
        pretrained_backbone: Use ImageNet pretrained weights
        freeze_backbone_stages: Number of backbone stages to freeze
        transformer_layers: Number of transformer encoder layers
        vam_layers: Number of VAM attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        d_model: int = 256,
        pretrained_backbone: bool = True,
        freeze_backbone_stages: int = 2,
        transformer_layers: int = 4,
        vam_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        
        print(f"\n{'='*60}")
        print("Initializing MAHT-Net")
        print(f"{'='*60}")
        
        # 1. CNN Backbone
        self.backbone = EfficientNetV2Backbone(
            pretrained=pretrained_backbone,
            freeze_stages=freeze_backbone_stages,
            out_channels=d_model
        )
        
        # 2. Transformer Bridge
        self.transformer = TransformerBridge(
            d_model=d_model,
            nhead=8,
            num_layers=transformer_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # 3. Vertebral Attention Module
        self.vam = VAM(
            d_model=d_model,
            num_keypoints=num_keypoints,
            nhead=8,
            num_layers=vam_layers,
            dropout=dropout
        )
        
        # 4. Multi-scale Decoder
        self.decoder = MultiScaleDecoder(
            num_keypoints=num_keypoints,
            d_model=d_model,
            skip_channels=[24, 48, 64],  # EfficientNetV2-S channels
            decoder_channels=128,
            dropout=dropout
        )
        
        print(f"{'='*60}")
        print(f"MAHT-Net initialized: {self._count_parameters():.2f}M parameters")
        print(f"{'='*60}\n")
    
    def _count_parameters(self) -> float:
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
                'keypoints': (B, K, 2) - extracted coordinates (inference only)
        """
        # 1. Extract multi-scale features
        f4, skip_features = self.backbone(x)  # f4: (B, d_model, 16, 16)
        
        # 2. Global context modeling
        global_features = self.transformer(f4)  # (B, d_model, 16, 16)
        
        # 3. Vertebral attention
        vam_output = self.vam(global_features)  # (B, K, d_model)
        
        # 4. Generate heatmaps
        heatmaps = self.decoder(vam_output, skip_features, global_features)
        
        return {'heatmaps': heatmaps}
    
    def extract_keypoints(
        self, 
        heatmaps: torch.Tensor,
        use_dark: bool = True
    ) -> torch.Tensor:
        """
        Extract keypoint coordinates from heatmaps.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            use_dark: Use DARK sub-pixel refinement
            
        Returns:
            keypoints: (B, K, 2) coordinates in pixel space
        """
        B, K, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)
        
        # Find argmax positions
        max_idx = heatmaps_flat.argmax(dim=2)  # (B, K)
        
        # Convert to (x, y) coordinates
        y = max_idx // W
        x = max_idx % W
        
        keypoints = torch.stack([x, y], dim=2).float()  # (B, K, 2)
        
        if use_dark:
            keypoints = self._dark_refinement(heatmaps, keypoints)
        
        return keypoints
    
    def _dark_refinement(
        self,
        heatmaps: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        DARK (Distribution-Aware coordinate Representation of Keypoints)
        Fits a 2D Gaussian to local region for sub-pixel accuracy.
        """
        B, K, H, W = heatmaps.shape
        
        refined = keypoints.clone()
        
        for b in range(B):
            for k in range(K):
                x, y = int(keypoints[b, k, 0]), int(keypoints[b, k, 1])
                
                # Skip if too close to border
                if x <= 0 or x >= W-1 or y <= 0 or y >= H-1:
                    continue
                
                # Get 3x3 neighborhood
                patch = heatmaps[b, k, y-1:y+2, x-1:x+2]
                
                if patch.max() <= 0:
                    continue
                
                # Taylor expansion for sub-pixel offset
                dx = (patch[1, 2] - patch[1, 0]) / 2
                dy = (patch[2, 1] - patch[0, 1]) / 2
                
                # Clamp offset to reasonable range
                dx = torch.clamp(dx, -0.5, 0.5)
                dy = torch.clamp(dy, -0.5, 0.5)
                
                refined[b, k, 0] = x + dx
                refined[b, k, 1] = y + dy
        
        return refined


def create_maht_net(view: str = 'AP', **kwargs) -> MAHTNet:
    """
    Factory function to create MAHT-Net for specific view.
    
    Args:
        view: 'AP' (20 keypoints) or 'LA' (22 keypoints)
        **kwargs: Additional arguments for MAHTNet
        
    Returns:
        Configured MAHTNet instance
    """
    num_keypoints = 20 if view.upper() == 'AP' else 22
    return MAHTNet(num_keypoints=num_keypoints, **kwargs)


# Register with model registry
def register_maht_net():
    """Register MAHT-Net with the model registry."""
    try:
        from .model_registry import ModelRegistry
        
        @ModelRegistry.register('maht-net')
        def _create(**kwargs):
            return create_maht_net(**kwargs)
        
        @ModelRegistry.register('maht-net-ap')
        def _create_ap(**kwargs):
            return create_maht_net(view='AP', **kwargs)
        
        @ModelRegistry.register('maht-net-la')
        def _create_la(**kwargs):
            return create_maht_net(view='LA', **kwargs)
            
    except ImportError:
        pass


# Test
if __name__ == "__main__":
    print("Testing MAHT-Net...")
    
    model = MAHTNet(num_keypoints=20)
    x = torch.randn(2, 3, 512, 512)
    
    output = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output heatmaps: {output['heatmaps'].shape}")
    
    # Test keypoint extraction
    kps = model.extract_keypoints(output['heatmaps'])
    print(f"Extracted keypoints: {kps.shape}")
    
    print("\n✅ MAHT-Net test passed!")
```

---

## Step 2.7: Components `__init__.py`

Create `models/components/__init__.py`:

```python
"""MAHT-Net architectural components."""

from .efficientnet_backbone import EfficientNetV2Backbone
from .transformer_bridge import TransformerBridge
from .vam import VAM
from .decoder import MultiScaleDecoder

__all__ = [
    'EfficientNetV2Backbone',
    'TransformerBridge',
    'VAM',
    'MultiScaleDecoder'
]
```

---

## Step 2.8: Testing the Full Model

Create test script `scripts/test_maht_net.py`:

```python
#!/usr/bin/env python3
"""Test the full MAHT-Net implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.maht_net import MAHTNet, create_maht_net


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("Testing MAHT-Net Forward Pass")
    print("=" * 60)
    
    # Test AP view (20 keypoints)
    print("\n1. Testing AP view (20 keypoints)...")
    model_ap = create_maht_net(view='AP')
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model_ap(x)
    
    assert output['heatmaps'].shape == (2, 20, 512, 512), \
        f"Expected (2, 20, 512, 512), got {output['heatmaps'].shape}"
    print("   ✓ AP forward pass OK")
    
    # Test LA view (22 keypoints)
    print("\n2. Testing LA view (22 keypoints)...")
    model_la = create_maht_net(view='LA')
    
    with torch.no_grad():
        output = model_la(x)
    
    assert output['heatmaps'].shape == (2, 22, 512, 512), \
        f"Expected (2, 22, 512, 512), got {output['heatmaps'].shape}"
    print("   ✓ LA forward pass OK")
    
    # Test keypoint extraction
    print("\n3. Testing keypoint extraction...")
    keypoints = model_ap.extract_keypoints(output['heatmaps'][:, :20])
    assert keypoints.shape == (2, 20, 2), \
        f"Expected (2, 20, 2), got {keypoints.shape}"
    print("   ✓ Keypoint extraction OK")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_memory_usage():
    """Test GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    print("\n" + "=" * 60)
    print("Testing GPU Memory Usage")
    print("=" * 60)
    
    device = torch.device('cuda')
    model = create_maht_net(view='AP').to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    x = torch.randn(4, 3, 512, 512, device=device)
    with torch.no_grad():
        output = model(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory (batch=4): {peak_memory:.2f} GB")
    
    if peak_memory > 15:
        print("⚠️  Warning: May exceed T4 GPU memory (15GB)")
    else:
        print("✓ Memory usage OK for Colab T4")


if __name__ == "__main__":
    test_forward_pass()
    test_memory_usage()
```

---

## Verification Commands

```bash
# 1. Create component files
mkdir -p models/components

# 2. Test each component
python -c "from models.components.efficientnet_backbone import EfficientNetV2Backbone; print('Backbone OK')"
python -c "from models.components.transformer_bridge import TransformerBridge; print('Transformer OK')"
python -c "from models.components.vam import VAM; print('VAM OK')"
python -c "from models.components.decoder import MultiScaleDecoder; print('Decoder OK')"

# 3. Test full model
python scripts/test_maht_net.py

# 4. Count parameters
python -c "from models.maht_net import MAHTNet; m = MAHTNet(); print(f'Parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')"
```

---

## Implementation Checklist

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| CNN Backbone | `components/efficientnet_backbone.py` | ⬜ | ~100 |
| Transformer Bridge | `components/transformer_bridge.py` | ⬜ | ~120 |
| VAM | `components/vam.py` | ⬜ | ~180 |
| Decoder | `components/decoder.py` | ⬜ | ~150 |
| Main Model | `maht_net.py` | ⬜ | ~200 |
| Components __init__ | `components/__init__.py` | ⬜ | ~10 |

**Total**: ~760 lines of PyTorch code

---

## Next Step

After implementing MAHT-Net, proceed to:
- [03_training_pipeline.md](03_training_pipeline.md) - Training on Google Colab

---

## Reference Documentation

- [01_cnn_backbone.md](../architecture/01_cnn_backbone.md) - Detailed backbone design
- [02_transformer_bridge.md](../architecture/02_transformer_bridge.md) - Transformer details
- [03_vertebral_attention_module.md](../architecture/03_vertebral_attention_module.md) - VAM design
- [04_multiscale_decoder.md](../architecture/04_multiscale_decoder.md) - Decoder architecture
- [05_dark_decoding.md](../architecture/05_dark_decoding.md) - Sub-pixel extraction
- [06_loss_functions.md](../architecture/06_loss_functions.md) - Loss function design

---

*Last Updated: February 2025*
