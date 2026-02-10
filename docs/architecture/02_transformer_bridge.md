# Component 2: Transformer Bridge

## Overview

The Transformer Bridge processes the high-level CNN features (F4) to model **global spatial relationships** across the entire image. This is critical for understanding vertebral structure, as each vertebra's position depends on its neighbors.

## Why a Transformer?

| Property | CNN Alone | CNN + Transformer |
|----------|-----------|-------------------|
| **Receptive field** | Local (limited) | Global (full image) |
| **Long-range dependencies** | Multiple layers needed | Single attention layer |
| **Positional awareness** | Implicit (stride patterns) | Explicit (position encoding) |
| **Vertebra relationships** | Learned indirectly | Directly modeled |

**Key insight**: CNNs excel at local feature extraction but struggle with global context. A vertebra at L1 should "know about" the vertebrae at L5 for consistent predictions.

## Architecture

```
F4 Features (16×16×256)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     PATCH EMBEDDING                           │
│  Flatten spatial dims: 16×16 → 256 tokens                    │
│  Each token: 256-dimensional                                  │
│  Add positional encoding (learnable 2D)                       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   TRANSFORMER ENCODER                         │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Layer 1: Multi-Head Self-Attention (8 heads)       │    │
│   │           + FFN (256 → 1024 → 256)                  │    │
│   │           + LayerNorm + Residual                    │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Layer 2: Multi-Head Self-Attention (8 heads)       │    │
│   │           + FFN (256 → 1024 → 256)                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Layer 3: Multi-Head Self-Attention (8 heads)       │    │
│   │           + FFN (256 → 1024 → 256)                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Layer 4: Multi-Head Self-Attention (8 heads)       │    │
│   │           + FFN (256 → 1024 → 256)                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     RESHAPE OUTPUT                            │
│  Unflatten: 256 tokens → 16×16 spatial                       │
│  Output: 16×16×256 (same shape as input)                     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Global Features (16×16×256) → to VAM
```

## Self-Attention Mechanism

### How Attention Works for Vertebrae

```
Query (Q): "What should this position attend to?"
Key (K):   "What information does this position have?"
Value (V): "What information to pass if attended?"

Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Example**: A token at the L4 position:
1. Creates a Query: "I'm looking for nearby vertebrae"
2. Computes similarity with all Keys (256 positions)
3. High attention to L3 and L5 positions
4. Aggregates their Values to understand vertebral context

### Multi-Head Attention

```
                Input (256 tokens × 256 dim)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
       ┌─────┐       ┌─────┐       ┌─────┐
       │Head1│       │Head2│  ...  │Head8│
       │32dim│       │32dim│       │32dim│
       └─────┘       └─────┘       └─────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
                  Concatenate
                        │
                  Linear (256×256)
                        │
                    Output
```

Each head can learn different relationships:
- **Head 1**: Adjacent vertebrae (L4 ↔ L5)
- **Head 2**: Symmetric corners (left ↔ right)
- **Head 3**: Superior-inferior relationships
- etc.

## Implementation

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial features.
    
    Unlike 1D positional encoding for sequences, this encodes
    both x and y positions independently.
    """
    
    def __init__(self, dim: int, height: int, width: int):
        """
        Args:
            dim: Embedding dimension (must be divisible by 4)
            height: Feature map height
            width: Feature map width
        """
        super().__init__()
        
        assert dim % 4 == 0, "Dimension must be divisible by 4"
        
        # Create position indices
        y_pos = torch.arange(height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1)
        
        # Compute sinusoidal encodings
        dim_quarter = dim // 4
        div_term = torch.exp(
            torch.arange(0, dim_quarter, 2) * (-math.log(10000.0) / dim_quarter)
        )
        
        pe = torch.zeros(height, width, dim)
        
        # Y position encoding (first half)
        pe[:, :, 0:dim_quarter:2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1:dim_quarter:2] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        # X position encoding (second half)
        pe[:, :, dim_quarter:dim_quarter*2:2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, dim_quarter+1:dim_quarter*2:2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        
        # Repeat for remaining dimensions
        pe[:, :, dim_quarter*2:dim_quarter*3] = pe[:, :, 0:dim_quarter]
        pe[:, :, dim_quarter*3:] = pe[:, :, dim_quarter:dim_quarter*2]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, H, W, D]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, H, W, D]
            
        Returns:
            x + positional encoding
        """
        return x + self.pe


class LearnablePositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding.
    
    Often works better than sinusoidal for small feature maps.
    """
    
    def __init__(self, dim: int, height: int, width: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, height, width, dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → + x → LayerNorm → FFN → + x
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth (drop path)
        self.drop_path = nn.Identity()  # Can add DropPath here
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D] where N = H*W
            
        Returns:
            Output tensor [B, N, D]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # FFN with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class TransformerBridge(nn.Module):
    """
    Transformer bridge between CNN backbone and VAM.
    
    Processes CNN features to model global spatial relationships
    across the entire image.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        feature_size: tuple = (16, 16)
    ):
        """
        Args:
            in_channels: Input feature channels from CNN
            dim: Transformer hidden dimension
            depth: Number of Transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            feature_size: Spatial size of input features (H, W)
        """
        super().__init__()
        
        self.feature_size = feature_size
        H, W = feature_size
        
        # Input projection (if channels don't match)
        self.input_proj = nn.Conv2d(in_channels, dim, 1) if in_channels != dim else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding2D(dim, H, W)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process CNN features through Transformer.
        
        Args:
            x: CNN features [B, C, H, W]
            
        Returns:
            Global features [B, C, H, W] (same shape as input)
        """
        B, C, H, W = x.shape
        
        # Project input channels
        x = self.input_proj(x)  # [B, dim, H, W]
        
        # Reshape to sequence: [B, H, W, dim]
        x = x.permute(0, 2, 3, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Flatten spatial dimensions: [B, H*W, dim]
        x = x.reshape(B, H * W, -1)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Reshape back to spatial: [B, dim, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return x


# Usage example
if __name__ == "__main__":
    # Create Transformer Bridge
    transformer = TransformerBridge(
        in_channels=256,
        dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        feature_size=(16, 16)
    )
    
    # Simulate CNN backbone output
    batch_size = 4
    f4 = torch.randn(batch_size, 256, 16, 16)
    
    # Forward pass
    global_features = transformer(f4)
    
    print(f"Input shape: {f4.shape}")
    print(f"Output shape: {global_features.shape}")
    print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
```

## Key Design Decisions

### 1. Number of Layers (Depth = 4)

| Depth | Pros | Cons |
|-------|------|------|
| 2 | Fast, fewer params | Limited global modeling |
| **4** | Good balance | **Recommended** |
| 6 | Better global context | Slower, more memory |
| 8 | Maximum capacity | Overkill for 16×16 |

**Decision**: 4 layers provide sufficient global modeling for a 16×16 feature map (256 tokens).

### 2. Number of Heads (8)

With dim=256 and 8 heads, each head has dimension 32.

**Rationale**: 8 heads allow learning diverse attention patterns:
- Spatial neighbors
- Same-vertebra corners
- Cross-vertebra relationships
- Symmetric patterns

### 3. Positional Encoding (Learnable)

**Decision**: Learnable positional encoding over sinusoidal.

**Rationale**:
- Feature map is small (16×16), so learnable parameters are manageable
- Can learn task-specific position importance
- Often works better for vision tasks

### 4. Pre-LayerNorm Architecture

We use Pre-LayerNorm (LayerNorm before attention/MLP) instead of Post-LayerNorm:

```
Pre-LayerNorm:  x → LayerNorm → Attn → + x → LayerNorm → MLP → + x
Post-LayerNorm: x → Attn → + x → LayerNorm → MLP → + x → LayerNorm
```

**Rationale**: Pre-LayerNorm trains more stably without learning rate warmup.

## Computational Cost

For input size 16×16×256 and batch size 8:

| Component | FLOPs | Memory |
|-----------|-------|--------|
| Attention (per layer) | ~33M | ~4MB |
| MLP (per layer) | ~33M | ~2MB |
| **Total (4 layers)** | **~264M** | **~24MB** |

This is very manageable for Colab.

## Attention Visualization

The attention patterns can be visualized to understand what the model learns:

```python
def visualize_attention(transformer, x):
    """Extract and visualize attention maps."""
    
    # Hook to capture attention weights
    attention_maps = []
    
    def hook(module, input, output):
        # output[1] contains attention weights
        attention_maps.append(output[1])
    
    # Register hooks
    hooks = []
    for block in transformer.blocks:
        h = block.attn.register_forward_hook(hook)
        hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = transformer(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return attention_maps  # List of [B, H, N, N] tensors
```

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
2. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words. ICLR. (ViT)
3. Carion, N., et al. (2020). End-to-End Object Detection with Transformers. ECCV. (DETR)
4. Liu, Z., et al. (2021). Swin Transformer. ICCV.
