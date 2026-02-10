"""
Component: Transformer Bridge
Models global spatial relationships in feature maps.

This lightweight transformer processes flattened CNN features (16×16 → 256 tokens)
through multi-head self-attention layers for global context modeling.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for spatial features.
    
    Creates separate row and column embeddings that are concatenated
    to form full positional encodings for each spatial location.
    """
    
    def __init__(self, d_model: int, height: int = 16, width: int = 16):
        super().__init__()
        
        # Learnable position embeddings
        self.row_embed = nn.Parameter(torch.randn(height, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(width, d_model // 2) * 0.02)
    
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


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Fixed sinusoidal 2D positional encoding (non-learnable).
    
    Uses sine/cosine functions at different frequencies for position encoding.
    Similar to the original Transformer but extended to 2D.
    """
    
    def __init__(self, d_model: int, height: int = 16, width: int = 16):
        super().__init__()
        
        self.d_model = d_model
        
        # Create position encoding matrix
        pe = torch.zeros(height * width, d_model)
        
        # Create position indices
        positions = torch.arange(height * width).float().unsqueeze(1)
        
        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, H*W, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding."""
        return x + self.pe[:, :x.size(1)]


class TransformerBridge(nn.Module):
    """
    Transformer encoder for global context modeling.
    
    Processes flattened CNN features (16×16 → 256 tokens) through
    multi-head self-attention layers. Essential for capturing
    long-range dependencies between vertebrae.
    
    Key features:
    - Pre-LN (Layer Normalization before attention) for stable training
    - Learnable 2D positional encoding
    - GELU activation in FFN
    
    Args:
        d_model: Transformer dimension (should match CNN output)
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        feature_size: Spatial size of input features (default 16×16)
        use_learnable_pe: Use learnable vs fixed positional encoding
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        feature_size: int = 16,
        use_learnable_pe: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.feature_size = feature_size
        
        # Positional encoding
        if use_learnable_pe:
            self.pos_encoding = PositionalEncoding2D(d_model, feature_size, feature_size)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding2D(d_model, feature_size, feature_size)
        
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
        
        print(f"  ✓ TransformerBridge: {num_layers} layers, {nhead} heads, d={d_model}")
    
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
    print("\n" + "="*60)
    print("Testing TransformerBridge")
    print("="*60)
    
    model = TransformerBridge(d_model=256, num_layers=4)
    x = torch.randn(2, 256, 16, 16)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Output: {out.shape}")
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    print("\n✓ TransformerBridge test passed!")


if __name__ == "__main__":
    test_transformer()
