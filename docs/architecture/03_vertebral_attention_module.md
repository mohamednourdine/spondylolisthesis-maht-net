# Component 3: Vertebral Attention Module (VAM)

## Overview

The **Vertebral Attention Module (VAM)** is our key architectural contribution. It uses learnable keypoint queries to attend to specific vertebral locations in the global features, incorporating anatomical priors about vertebral structure.

## Why VAM?

Traditional keypoint detection approaches treat all keypoints independently. However, vertebral corners have strong anatomical relationships:

- **Spatial ordering**: L1 is always above L2, which is always above L3, etc.
- **Local structure**: 4 corners of a vertebra form a roughly rectangular shape
- **Symmetry**: Left and right corners at the same level should be roughly symmetric (in AP view)

VAM encodes these priors through **anatomical position encoding** and **structured attention patterns**.

## Architecture

```
Global Features (16×16×256)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   KEYPOINT QUERIES                            │
│                                                               │
│   Q = [q₁, q₂, ..., qₖ]  (K learnable queries)               │
│   K = 20 (AP view) or K = 22 (LA view)                       │
│                                                               │
│   Each query represents one keypoint:                         │
│   q₁ = L1-upper-left, q₂ = L1-upper-right, ...               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│              ANATOMICAL POSITION ENCODING                     │
│                                                               │
│   Add anatomical priors to queries:                           │
│   - Vertebra level (L1, L2, L3, L4, L5, S1)                   │
│   - Corner type (upper-left, upper-right, lower-left, ...)    │
│   - Expected relative position                                │
│                                                               │
│   q_i' = q_i + pos_vertebra(i) + pos_corner(i)               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│               CROSS-ATTENTION LAYERS                          │
│                                                               │
│   For each layer (×3):                                        │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  1. Cross-Attention:                                 │    │
│   │     Q = keypoint queries                             │    │
│   │     K, V = flattened global features                 │    │
│   │     → Each query attends to relevant image regions   │    │
│   │                                                      │    │
│   │  2. Self-Attention (optional):                       │    │
│   │     Queries attend to each other                     │    │
│   │     → Model inter-keypoint relationships             │    │
│   │                                                      │    │
│   │  3. FFN + Residual + LayerNorm                       │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Attended Features (K × 256)
        │
        ▼
[To Decoder: reshaped and projected]
```

## Keypoint Query Design

### Query Mapping

For **AP View** (20 keypoints = 5 vertebrae × 4 corners):

| Query Index | Vertebra | Corner | Description |
|-------------|----------|--------|-------------|
| 0 | L1 | Upper-Left | Top-left corner of L1 |
| 1 | L1 | Upper-Right | Top-right corner of L1 |
| 2 | L1 | Lower-Left | Bottom-left corner of L1 |
| 3 | L1 | Lower-Right | Bottom-right corner of L1 |
| 4 | L2 | Upper-Left | ... |
| ... | ... | ... | ... |
| 19 | L5 | Lower-Right | Bottom-right corner of L5 |

For **LA View** (22 keypoints = 5 vertebrae × 4 corners + 2 S1):

| Query Index | Vertebra | Corner |
|-------------|----------|--------|
| 0-19 | L1-L5 | Same as AP |
| 20 | S1 | Upper-Left |
| 21 | S1 | Upper-Right |

### Anatomical Position Encoding

```python
# Vertebra level encoding (6 levels: L1-L5 + S1)
vertebra_encoding = nn.Embedding(6, dim // 2)

# Corner type encoding (4 types: UL, UR, LL, LR)  
corner_encoding = nn.Embedding(4, dim // 2)

# Combined anatomical encoding
anatomical_pos[i] = concat(
    vertebra_encoding[vertebra_of(i)],
    corner_encoding[corner_of(i)]
)
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicalPositionEncoding(nn.Module):
    """
    Encodes anatomical priors into keypoint queries.
    
    Encodes:
    - Which vertebra (L1-L5, S1)
    - Which corner (upper-left, upper-right, lower-left, lower-right)
    """
    
    def __init__(self, num_keypoints: int, dim: int, view: str = 'AP'):
        """
        Args:
            num_keypoints: Number of keypoints (20 for AP, 22 for LA)
            dim: Embedding dimension
            view: 'AP' or 'LA'
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.view = view
        
        # Vertebra level encoding (L1=0, L2=1, ..., L5=4, S1=5)
        num_vertebrae = 6 if view == 'LA' else 5
        self.vertebra_encoding = nn.Embedding(num_vertebrae, dim // 2)
        
        # Corner type encoding (UL=0, UR=1, LL=2, LR=3)
        self.corner_encoding = nn.Embedding(4, dim // 2)
        
        # Create lookup tables
        self._create_lookup_tables()
        
    def _create_lookup_tables(self):
        """Create vertebra and corner indices for each keypoint."""
        
        vertebra_indices = []
        corner_indices = []
        
        # Standard vertebrae (L1-L5)
        for v in range(5):  # L1-L5
            for c in range(4):  # UL, UR, LL, LR
                vertebra_indices.append(v)
                corner_indices.append(c)
        
        # S1 (only for LA view)
        if self.view == 'LA':
            vertebra_indices.extend([5, 5])  # S1
            corner_indices.extend([0, 1])    # Only upper corners
        
        self.register_buffer('vertebra_indices', torch.tensor(vertebra_indices))
        self.register_buffer('corner_indices', torch.tensor(corner_indices))
        
    def forward(self) -> torch.Tensor:
        """
        Generate anatomical position encodings.
        
        Returns:
            Tensor of shape [K, dim] with anatomical encodings
        """
        vertebra_emb = self.vertebra_encoding(self.vertebra_indices)  # [K, dim//2]
        corner_emb = self.corner_encoding(self.corner_indices)        # [K, dim//2]
        
        return torch.cat([vertebra_emb, corner_emb], dim=-1)  # [K, dim]


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer where queries attend to image features.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Keypoint queries [B, K, D]
            features: Global features [B, N, D] (N = H*W)
            
        Returns:
            Updated queries [B, K, D]
        """
        # Cross-attention
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(features)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        queries = queries + attn_out
        
        # FFN
        queries = queries + self.ffn(self.norm_ffn(queries))
        
        return queries


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer for queries to model inter-keypoint relationships.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Keypoint queries [B, K, D]
            
        Returns:
            Updated queries [B, K, D]
        """
        q_norm = self.norm(queries)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        return queries + attn_out


class VertebralAttentionModule(nn.Module):
    """
    Vertebral Attention Module (VAM) - Novel contribution.
    
    Uses learnable keypoint queries with anatomical priors to
    attend to vertebral corner locations in the image features.
    """
    
    def __init__(
        self,
        num_keypoints: int,
        dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        view: str = 'AP',
        use_self_attention: bool = True
    ):
        """
        Args:
            num_keypoints: Number of keypoints (20 for AP, 22 for LA)
            dim: Feature dimension
            depth: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            view: 'AP' or 'LA' (affects anatomical encoding)
            use_self_attention: Include self-attention between queries
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.dim = dim
        self.use_self_attention = use_self_attention
        
        # Learnable keypoint queries
        self.keypoint_queries = nn.Parameter(torch.randn(num_keypoints, dim) * 0.02)
        
        # Anatomical position encoding
        self.anatomical_pe = AnatomicalPositionEncoding(num_keypoints, dim, view)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        # Self-attention layers (optional)
        if use_self_attention:
            self.self_attn_layers = nn.ModuleList([
                SelfAttentionLayer(dim, num_heads, dropout)
                for _ in range(depth)
            ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        global_features: torch.Tensor,
        multi_scale_features: dict = None
    ) -> torch.Tensor:
        """
        Process global features with keypoint queries.
        
        Args:
            global_features: Transformer output [B, C, H, W]
            multi_scale_features: Dict of CNN features (optional, for future use)
            
        Returns:
            Attended keypoint features [B, K, D]
        """
        B, C, H, W = global_features.shape
        
        # Flatten spatial dimensions
        features = global_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Initialize queries with anatomical encoding
        # Expand queries for batch
        queries = self.keypoint_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        
        # Add anatomical position encoding
        anatomical_pe = self.anatomical_pe().unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        queries = queries + anatomical_pe
        
        # Apply attention layers
        for i, cross_attn in enumerate(self.cross_attn_layers):
            # Cross-attention: queries attend to features
            queries = cross_attn(queries, features)
            
            # Self-attention: queries attend to each other
            if self.use_self_attention:
                queries = self.self_attn_layers[i](queries)
        
        # Final normalization
        queries = self.norm(queries)
        
        return queries


class VAMWithBiasMatrix(VertebralAttentionModule):
    """
    VAM variant with explicit anatomical bias matrix.
    
    Adds learnable bias to attention weights based on anatomical
    relationships between keypoints.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        K = self.num_keypoints
        
        # Learnable attention bias between keypoints
        # Initialized based on anatomical relationships
        self.attn_bias = nn.Parameter(self._init_anatomical_bias(K))
        
    def _init_anatomical_bias(self, K: int) -> torch.Tensor:
        """
        Initialize bias matrix based on anatomical relationships.
        
        Same vertebra corners → high bias
        Adjacent vertebra corners → medium bias
        Far vertebra corners → low/negative bias
        """
        bias = torch.zeros(K, K)
        
        for i in range(K):
            for j in range(K):
                vert_i = i // 4  # Vertebra index
                vert_j = j // 4
                corner_i = i % 4  # Corner index
                corner_j = j % 4
                
                # Same vertebra
                if vert_i == vert_j:
                    bias[i, j] = 0.5  # High affinity
                # Adjacent vertebrae
                elif abs(vert_i - vert_j) == 1:
                    bias[i, j] = 0.2  # Medium affinity
                    # Same corner type on adjacent vertebrae
                    if corner_i == corner_j:
                        bias[i, j] = 0.3
                # Distant vertebrae
                else:
                    bias[i, j] = -0.1  # Low affinity
        
        return bias


# Usage example
if __name__ == "__main__":
    # Create VAM for AP view
    vam_ap = VertebralAttentionModule(
        num_keypoints=20,
        dim=256,
        depth=3,
        num_heads=8,
        view='AP'
    )
    
    # Create VAM for LA view
    vam_la = VertebralAttentionModule(
        num_keypoints=22,
        dim=256,
        depth=3,
        num_heads=8,
        view='LA'
    )
    
    # Simulate global features from Transformer
    batch_size = 4
    global_features = torch.randn(batch_size, 256, 16, 16)
    
    # Forward pass
    ap_queries = vam_ap(global_features)
    la_queries = vam_la(global_features)
    
    print(f"AP queries shape: {ap_queries.shape}")  # [4, 20, 256]
    print(f"LA queries shape: {la_queries.shape}")  # [4, 22, 256]
    print(f"VAM AP parameters: {sum(p.numel() for p in vam_ap.parameters()):,}")
```

## Attention Visualization

Visualizing what each keypoint query attends to:

```python
def visualize_vam_attention(vam, global_features, query_idx=0):
    """
    Visualize attention weights for a specific keypoint query.
    
    Args:
        vam: VertebralAttentionModule
        global_features: [1, C, H, W]
        query_idx: Which keypoint to visualize
    """
    import matplotlib.pyplot as plt
    
    # Hook to capture attention
    attention_weights = []
    
    def hook(module, input, output):
        attention_weights.append(output[1])  # [B, K, N]
    
    # Register hook on first cross-attention
    handle = vam.cross_attn_layers[0].cross_attn.register_forward_hook(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = vam(global_features)
    
    handle.remove()
    
    # Get attention for specific query
    attn = attention_weights[0][0, query_idx]  # [N]
    attn = attn.reshape(16, 16).numpy()
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(attn, cmap='hot')
    plt.title(f'Attention for keypoint {query_idx}')
    plt.colorbar()
    plt.show()
```

## Key Design Decisions

### 1. Learnable Queries vs. Fixed

**Decision**: Learnable queries initialized randomly.

**Rationale**: 
- Queries learn to specialize for their assigned keypoint
- Anatomical encoding provides structure
- More flexible than fixed queries

### 2. Anatomical Position Encoding

**Decision**: Separate vertebra-level and corner-type embeddings.

**Rationale**:
- Encodes hierarchical structure (vertebra > corner)
- Allows model to learn relationships at both levels
- Concatenation preserves both components

### 3. Self-Attention Between Queries

**Decision**: Include self-attention (optional).

**Rationale**:
- Queries can share information about their predictions
- Enforces consistency (e.g., if L4-lower detected, L5-upper should be nearby)
- Can be disabled if overfitting

### 4. Number of Layers (Depth = 3)

**Decision**: 3 cross-attention layers.

**Rationale**:
- First layer: Coarse localization
- Second layer: Refinement
- Third layer: Fine-tuning
- More layers showed diminishing returns in experiments

## Comparison with Alternatives

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Direct regression** | CNN → FC → coordinates | Simple | Poor accuracy |
| **Heatmap-only** | CNN → Conv → heatmaps | Proven effective | No explicit relationships |
| **DETR-style** | Queries + cross-attention | Flexible | No anatomical priors |
| **VAM (Ours)** | Queries + anatomical encoding | Domain-specific | Requires design effort |

## References

1. Carion, N., et al. (2020). End-to-End Object Detection with Transformers. ECCV. (DETR)
2. Mao, J., et al. (2022). PoseFormer: Query-Based Transformer for Human Pose Estimation. (Query-based keypoints)
3. Li, Y., et al. (2021). TokenPose: Learning Keypoint Tokens for Human Pose Estimation. ICCV.
