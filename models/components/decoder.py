"""
Component: Multi-scale Decoder for Heatmap Generation

This module contains decoder architectures for generating keypoint heatmaps
from CNN backbone features.

Two versions:
1. SimpleDecoder - Phase 1: Direct backbone → heatmap (no VAM)
2. MultiScaleDecoder - Phase 2+: With VAM integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and optional skip connection.
    
    Architecture:
        ConvTranspose2d (2x upsample) → Concat skip → Conv → BN → ReLU → Conv → BN → ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_skip: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        # Upsample (2x) via transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=4, stride=2, padding=1
        )
        
        # Skip connection projection (to match channels)
        if use_skip and skip_channels > 0:
            self.skip_proj = nn.Conv2d(skip_channels, in_channels, kernel_size=1)
            fusion_channels = in_channels * 2  # Concatenated
        else:
            self.skip_proj = None
            fusion_channels = in_channels
        
        # Fusion convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(fusion_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Feature from previous decoder level (B, C, H, W)
            skip: Skip connection from encoder (B, C_skip, H*2, W*2)
        """
        x = self.upsample(x)
        
        if self.use_skip and skip is not None and self.skip_proj is not None:
            # Project skip to match channels
            skip = self.skip_proj(skip)
            
            # Resize skip to match x's spatial dimensions (x is the upsampled tensor)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class SimpleDecoder(nn.Module):
    """
    Simple decoder for Phase 1 (no VAM integration).
    
    Takes backbone F4 features and skip connections, produces heatmaps.
    This is a baseline decoder architecture without the vertebral attention module.
    
    Args:
        num_keypoints: Number of output keypoints (20 for AP, 22 for LA)
        in_channels: Input channels from backbone F4 (256)
        skip_channels: Channel dimensions for F1, F2, F3 skips
        decoder_channels: Base decoder channel dimension (128)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        in_channels: int = 256,
        skip_channels: List[int] = [48, 64, 128],  # F1, F2, F3
        decoder_channels: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Initial projection from backbone F4
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (progressively upsample with skip connections)
        # Input: 16×16 → Output: 512×512 (5 upsampling steps)
        
        # 16→32 (no skip, just upsample)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # 32→64 with F3 skip (128 channels at 32×32)
        self.block2 = DecoderBlock(decoder_channels, skip_channels[2], decoder_channels, dropout=dropout)
        
        # 64→128 with F2 skip (64 channels at 64×64)  
        self.block3 = DecoderBlock(decoder_channels, skip_channels[1], decoder_channels // 2, dropout=dropout)
        
        # 128→256 with F1 skip (48 channels at 128×128)
        self.block4 = DecoderBlock(decoder_channels // 2, skip_channels[0], decoder_channels // 4, dropout=dropout)
        
        # 256→512 final upsample (no skip)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Output head: produce K heatmaps
        self.output_head = nn.Conv2d(decoder_channels // 4, num_keypoints, kernel_size=1)
        
        print(f"  ✓ SimpleDecoder: {num_keypoints} output channels")
    
    def forward(
        self,
        f4: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate heatmaps from backbone features.
        
        Args:
            f4: High-level features (B, C, 16, 16) from backbone
            skip_features: [F1, F2, F3] from backbone at different resolutions
        
        Returns:
            heatmaps: (B, K, 512, 512) keypoint heatmaps
        """
        F1, F2, F3 = skip_features
        
        # Project input
        x = self.input_proj(f4)  # (B, decoder_channels, 16, 16)
        
        # Decode with skip connections
        x = self.up1(x)           # 16→32
        x = self.block2(x, F3)    # 32→64
        x = self.block3(x, F2)    # 64→128
        x = self.block4(x, F1)    # 128→256
        x = self.final_up(x)      # 256→512
        
        # Output heatmaps
        heatmaps = self.output_head(x)  # (B, K, 512, 512)
        
        return heatmaps


class MultiScaleDecoder(nn.Module):
    """
    Full decoder with VAM integration (for Phase 2+).
    
    Takes VAM-attended features, skip connections, and global features
    to produce high-resolution heatmaps.
    
    Args:
        num_keypoints: Number of output keypoints (20 for AP, 22 for LA)
        d_model: VAM feature dimension (256)
        skip_channels: Channel dimensions for F1, F2, F3 skips
        decoder_channels: Base decoder channel dimension (128)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        d_model: int = 256,
        skip_channels: List[int] = [48, 64, 128],
        decoder_channels: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        
        # Project VAM output to spatial feature map
        # VAM output: (B, K, d_model) → (B, decoder_channels, 16, 16)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, decoder_channels * 16 * 16),
            nn.ReLU()
        )
        
        # Additional projection for global features
        self.global_proj = nn.Conv2d(d_model, decoder_channels, kernel_size=1)
        
        # Decoder blocks (same as SimpleDecoder)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = DecoderBlock(decoder_channels, skip_channels[2], decoder_channels, dropout=dropout)
        self.block3 = DecoderBlock(decoder_channels, skip_channels[1], decoder_channels // 2, dropout=dropout)
        self.block4 = DecoderBlock(decoder_channels // 2, skip_channels[0], decoder_channels // 4, dropout=dropout)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.output_head = nn.Conv2d(decoder_channels // 4, num_keypoints, 1)
        
        print(f"  ✓ MultiScaleDecoder: {num_keypoints} output channels (VAM-enabled)")
    
    def forward(
        self,
        vam_output: torch.Tensor,
        skip_features: List[torch.Tensor],
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate heatmaps from VAM output and skip features.
        
        Args:
            vam_output: (B, K, d_model) attended features from VAM
            skip_features: [F1, F2, F3] from backbone
            global_features: (B, d_model, H, W) from Transformer
        
        Returns:
            heatmaps: (B, K, 512, 512)
        """
        B, K, C = vam_output.shape
        F1, F2, F3 = skip_features
        
        # Mean-pool VAM features across keypoints for spatial seed
        vam_pooled = vam_output.mean(dim=1)  # (B, d_model)
        
        # Project to spatial
        x = self.input_proj(vam_pooled)  # (B, decoder_channels * 256)
        x = x.view(B, -1, 16, 16)  # (B, decoder_channels, 16, 16)
        
        # Process global features
        global_feat = self.global_proj(global_features)
        global_feat = F.adaptive_avg_pool2d(global_feat, (16, 16))
        
        # Combine VAM and global features
        x = torch.cat([x, global_feat], dim=1)
        x = self.initial_conv(x)
        
        # Decode with skip connections
        x = self.up1(x)
        x = self.block2(x, F3)
        x = self.block3(x, F2)
        x = self.block4(x, F1)
        x = self.final_up(x)
        
        # Output heatmaps
        heatmaps = self.output_head(x)
        
        return heatmaps


def test_simple_decoder():
    """Test the simple decoder (Phase 1)."""
    print("\nTesting SimpleDecoder...")
    print("-" * 50)
    
    decoder = SimpleDecoder(num_keypoints=20, in_channels=256)
    
    # Simulated backbone outputs
    f4 = torch.randn(2, 256, 16, 16)
    f1 = torch.randn(2, 48, 128, 128)
    f2 = torch.randn(2, 64, 64, 64)
    f3 = torch.randn(2, 128, 32, 32)
    
    with torch.no_grad():
        heatmaps = decoder(f4, [f1, f2, f3])
    
    print(f"\nInput F4: {f4.shape}")
    print(f"Skip F1: {f1.shape}")
    print(f"Skip F2: {f2.shape}")
    print(f"Skip F3: {f3.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    
    assert heatmaps.shape == (2, 20, 512, 512), f"Expected (2, 20, 512, 512), got {heatmaps.shape}"
    print("\n✓ SimpleDecoder test passed!")
    return True


def test_multiscale_decoder():
    """Test the full decoder (Phase 2+)."""
    print("\nTesting MultiScaleDecoder...")
    print("-" * 50)
    
    decoder = MultiScaleDecoder(num_keypoints=20, d_model=256)
    
    # Simulated inputs
    vam_out = torch.randn(2, 20, 256)
    f1 = torch.randn(2, 48, 128, 128)
    f2 = torch.randn(2, 64, 64, 64)
    f3 = torch.randn(2, 128, 32, 32)
    global_feat = torch.randn(2, 256, 16, 16)
    
    with torch.no_grad():
        heatmaps = decoder(vam_out, [f1, f2, f3], global_feat)
    
    print(f"\nVAM output: {vam_out.shape}")
    print(f"Global features: {global_feat.shape}")
    print(f"Output heatmaps: {heatmaps.shape}")
    
    assert heatmaps.shape == (2, 20, 512, 512)
    print("\n✓ MultiScaleDecoder test passed!")
    return True


if __name__ == "__main__":
    test_simple_decoder()
    test_multiscale_decoder()
