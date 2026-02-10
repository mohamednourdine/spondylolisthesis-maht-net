"""
Component: Vertebral Attention Module (VAM)
Our key architectural contribution - anatomically-aware keypoint attention.

The VAM uses learnable keypoint queries with anatomical priors to attend
to relevant positions in the global feature map. This encodes domain
knowledge about vertebral structure directly into the architecture.

Key innovations:
1. Learnable keypoint queries representing each corner
2. Anatomical position encoding (vertebra level, edge type, corner type)
3. Cross-attention: queries attend to image features
4. Self-attention: queries attend to each other for consistency
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
    - Edge type (upper, lower)
    - Corner type (left, right)
    - Continuous normalized y-position
    
    This encoding provides strong inductive bias about the expected
    spatial relationships between keypoints.
    
    Args:
        d_model: Output embedding dimension
        num_vertebrae: Number of vertebra levels (6 for L1-L5+S1)
        num_edges: Edge types (2: upper, lower)
        num_corners: Corner types (2: left, right)
    """
    
    def __init__(
        self,
        d_model: int,
        num_vertebrae: int = 6,
        num_edges: int = 2,
        num_corners: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Learnable embeddings for anatomical structure
        self.vertebra_embed = nn.Embedding(num_vertebrae, d_model // 4)
        self.edge_embed = nn.Embedding(num_edges, d_model // 4)  # upper/lower
        self.corner_embed = nn.Embedding(num_corners, d_model // 4)  # left/right
        
        # Continuous position embedding (normalized y-position)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small values."""
        for embed in [self.vertebra_embed, self.edge_embed, self.corner_embed]:
            nn.init.normal_(embed.weight, std=0.02)
    
    def forward(
        self,
        vertebra_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        corner_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
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


class VertebralAttentionModule(nn.Module):
    """
    Vertebral Attention Module (VAM).
    
    Uses learnable keypoint queries with anatomical priors to attend
    to relevant positions in the global feature map.
    
    The module operates in two stages:
    1. Cross-attention: Each keypoint query attends to image features
       to localize "where in the image is this corner?"
    2. Self-attention: Keypoint queries attend to each other to
       enforce anatomical consistency between corners
    
    Args:
        d_model: Feature dimension
        num_keypoints: Number of keypoints (20 for AP, 22 for LA)
        nhead: Number of attention heads
        num_layers: Number of cross+self attention layer pairs
        dropout: Dropout rate
        view: 'AP' or 'LA' for view-specific anatomical encoding
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_keypoints: int = 20,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        view: str = 'AP'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_keypoints = num_keypoints
        self.num_layers = num_layers
        self.view = view
        
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
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        print(f"  ✓ VAM: {num_keypoints} keypoints, {num_layers} layers, view={view}")
    
    def _register_anatomical_indices(self, num_keypoints: int):
        """
        Create anatomical index tensors for position encoding.
        
        BUU-LSPINE keypoint organization:
        - Each row has 2 corners (left, right)
        - Rows alternate: upper edge, lower edge
        - AP: 20 keypoints = 5 vertebrae × 4 corners
        - LA: 22 keypoints = 5 vertebrae × 4 corners + 2 S1 corners
        """
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
            positions.append(row / max(num_rows - 1, 1))  # normalized position
            
            # Right corner
            vertebra_ids.append(vertebra)
            edge_ids.append(edge)
            corner_ids.append(1)  # right
            positions.append(row / max(num_rows - 1, 1))
        
        self.register_buffer('vertebra_ids', torch.tensor(vertebra_ids))
        self.register_buffer('edge_ids', torch.tensor(edge_ids))
        self.register_buffer('corner_ids', torch.tensor(corner_ids))
        self.register_buffer('positions', torch.tensor(positions).unsqueeze(-1).float())
    
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
            attn_out, _ = self.cross_attn_layers[i](q, memory, memory)
            queries = queries + attn_out
            
            # Self-attention: queries attend to each other
            q = self.norm2[i](queries)
            attn_out, _ = self.self_attn_layers[i](q, q, q)
            queries = queries + attn_out
            
            # FFN
            queries = queries + self.ffn_layers[i](self.norm3[i](queries))
        
        # Output projection
        output = self.output_proj(queries)
        
        return output
    
    def get_attention_weights(self, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Returns:
            cross_attn_weights: (B, K, H*W) - where each keypoint attends
            self_attn_weights: (B, K, K) - keypoint relationships
        """
        B, C, H, W = global_features.shape
        memory = global_features.flatten(2).permute(0, 2, 1)
        
        anatomical_pos = self.anatomical_encoding(
            self.vertebra_ids, self.edge_ids, 
            self.corner_ids, self.positions
        )
        queries = self.keypoint_queries + anatomical_pos
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        
        # Get attention from last layer
        q = self.norm1[-1](queries)
        _, cross_weights = self.cross_attn_layers[-1](q, memory, memory, need_weights=True)
        
        q = self.norm2[-1](queries)
        _, self_weights = self.self_attn_layers[-1](q, q, q, need_weights=True)
        
        return cross_weights, self_weights


# Alias for backward compatibility
VAM = VertebralAttentionModule


def test_vam():
    """Test the VAM module."""
    print("\n" + "="*60)
    print("Testing Vertebral Attention Module (VAM)")
    print("="*60)
    
    # Test AP view
    print("\n1. Testing AP view (20 keypoints)...")
    model_ap = VAM(d_model=256, num_keypoints=20, num_layers=3, view='AP')
    x = torch.randn(2, 256, 16, 16)
    
    with torch.no_grad():
        out_ap = model_ap(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out_ap.shape}")
    assert out_ap.shape == (2, 20, 256), f"Wrong shape: {out_ap.shape}"
    print("✓ AP VAM passed")
    
    # Test LA view
    print("\n2. Testing LA view (22 keypoints)...")
    model_la = VAM(d_model=256, num_keypoints=22, num_layers=3, view='LA')
    
    with torch.no_grad():
        out_la = model_la(x)
    
    print(f"Output: {out_la.shape}")
    assert out_la.shape == (2, 22, 256), f"Wrong shape: {out_la.shape}"
    print("✓ LA VAM passed")
    
    # Test attention visualization
    print("\n3. Testing attention weights...")
    cross_attn, self_attn = model_ap.get_attention_weights(x)
    print(f"Cross-attention: {cross_attn.shape}")  # (B, K, H*W)
    print(f"Self-attention: {self_attn.shape}")    # (B, K, K)
    print("✓ Attention weights passed")
    
    # Count parameters
    params = sum(p.numel() for p in model_ap.parameters()) / 1e6
    print(f"\nVAM parameters: {params:.2f}M")
    
    print("\n✓ All VAM tests passed!")


if __name__ == "__main__":
    test_vam()
