"""
ResNet-50 backbone with simple decoder for heatmap-based keypoint detection.
Based on SimpleBaseline (Microsoft, ECCV 2018).

Architecture:
    - ResNet-50 encoder (pretrained on ImageNet)
    - Simple upsampling decoder (deconv blocks)
    - Output: heatmaps at full resolution (512×512×40)

Note: No FPN or attention - those are reserved for MAHT-Net.
This model tests the hypothesis: "Do pretrained weights help?"
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from typing import List, Optional


class DeconvBlock(nn.Module):
    """
    Deconvolution block: ConvTranspose2d → BatchNorm → ReLU
    Doubles spatial resolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResNetHeatmap(nn.Module):
    """
    ResNet-50 backbone with simple deconv decoder for heatmap regression.
    
    This architecture follows the SimpleBaseline approach (ECCV 2018) which
    achieved state-of-the-art results on human pose estimation with a simple
    design: pretrained backbone + deconv upsampling.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        num_keypoints: Number of output heatmap channels (40 = 10 vertebrae × 4 corners)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone_layers: Number of backbone layers to freeze (0-4)
            0 = freeze nothing
            1 = freeze conv1, bn1
            2 = freeze conv1, bn1, layer1
            3 = freeze conv1, bn1, layer1, layer2
            4 = freeze conv1, bn1, layer1, layer2, layer3
        dropout_rate: Dropout rate for decoder (default: 0.3)
        decoder_channels: Number of channels in decoder (default: 256)
    
    Input: [B, 3, 512, 512]
    Output: [B, 40, 512, 512] (heatmaps, no activation)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 40,
        pretrained: bool = True,
        freeze_backbone_layers: int = 2,
        dropout_rate: float = 0.3,
        decoder_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.pretrained = pretrained
        self.freeze_backbone_layers = freeze_backbone_layers
        
        # Load pretrained ResNet-50
        if pretrained:
            print("   Loading ResNet-50 with ImageNet pretrained weights...")
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            print("   Loading ResNet-50 without pretrained weights...")
            self.backbone = models.resnet50(weights=None)
        
        # Remove FC and avgpool (we need spatial features)
        # We'll manually call backbone layers in forward()
        
        # Freeze early layers if specified
        self._freeze_layers(freeze_backbone_layers)
        
        # Simple decoder: 5 deconv blocks to go from 16×16 to 512×512
        # ResNet output at layer4: 2048 channels, 16×16 resolution (for 512×512 input)
        self.decoder = nn.Sequential(
            DeconvBlock(2048, decoder_channels, dropout_rate=dropout_rate),  # 16→32
            DeconvBlock(decoder_channels, decoder_channels, dropout_rate=dropout_rate * 0.67),  # 32→64
            DeconvBlock(decoder_channels, decoder_channels, dropout_rate=dropout_rate * 0.33),  # 64→128
            DeconvBlock(decoder_channels, decoder_channels, dropout_rate=0.0),  # 128→256
            DeconvBlock(decoder_channels, decoder_channels, dropout_rate=0.0),  # 256→512
        )
        
        # Final 1×1 conv to get heatmaps (no activation - raw outputs like UNet)
        self.head = nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1)
        
        # Initialize decoder weights
        self._init_decoder_weights()
        
        # Print model info
        self._print_model_info()
    
    def _freeze_layers(self, num_layers: int):
        """Freeze early backbone layers."""
        if num_layers <= 0:
            return
        
        frozen_params = 0
        
        # Always freeze conv1 and bn1 if num_layers >= 1
        if num_layers >= 1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze layer1 if num_layers >= 2
        if num_layers >= 2:
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze layer2 if num_layers >= 3
        if num_layers >= 3:
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze layer3 if num_layers >= 4
        if num_layers >= 4:
            for param in self.backbone.layer3.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"   Frozen {frozen_params / 1e6:.2f}M backbone parameters (layers 0-{num_layers})")
    
    def _init_decoder_weights(self):
        """Initialize decoder weights using Kaiming initialization."""
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize head with small weights
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0)
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"   Frozen parameters: {frozen_params / 1e6:.2f}M")
        print(f"   Backbone: {backbone_params / 1e6:.2f}M | Decoder: {decoder_params / 1e6:.2f}M | Head: {head_params / 1e6:.2f}M")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, 512, 512]
            
        Returns:
            Heatmaps [B, 40, 512, 512]
        """
        # Backbone feature extraction (stop before avgpool/fc)
        x = self.backbone.conv1(x)      # 64, 256×256
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)    # 64, 128×128
        
        x = self.backbone.layer1(x)     # 256, 128×128
        x = self.backbone.layer2(x)     # 512, 64×64
        x = self.backbone.layer3(x)     # 1024, 32×32
        x = self.backbone.layer4(x)     # 2048, 16×16
        
        # Decoder: upsample back to 512×512
        x = self.decoder(x)             # 256, 512×512
        
        # Heatmap prediction (no activation - raw outputs)
        heatmaps = self.head(x)         # 40, 512×512
        
        return heatmaps
    
    def get_parameter_groups(self, backbone_lr: float, decoder_lr: float):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            backbone_lr: Learning rate for backbone (typically smaller)
            decoder_lr: Learning rate for decoder and head
            
        Returns:
            List of parameter group dicts for optimizer
        """
        # Backbone parameters (only trainable ones)
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        
        # Decoder parameters
        decoder_params = list(self.decoder.parameters())
        
        # Head parameters
        head_params = list(self.head.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'},
            {'params': head_params, 'lr': decoder_lr, 'name': 'head'},
        ]
        
        return param_groups
    
    def predict_with_uncertainty(self, x, n_samples: int = 10):
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


def create_resnet_heatmap(**kwargs) -> ResNetHeatmap:
    """
    Factory function to create ResNetHeatmap model.
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_keypoints: Number of output channels (default: 40)
        pretrained: Use ImageNet weights (default: True)
        freeze_backbone_layers: Layers to freeze (default: 2)
        dropout_rate: Decoder dropout (default: 0.3)
        
    Returns:
        ResNetHeatmap model instance
    """
    return ResNetHeatmap(**kwargs)


# For testing
if __name__ == '__main__':
    print("Testing ResNetHeatmap model...")
    
    # Create model
    model = ResNetHeatmap(
        in_channels=3,
        num_keypoints=40,
        pretrained=True,
        freeze_backbone_layers=2,
        dropout_rate=0.3
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    print(f"\nInput shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # Test parameter groups
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, decoder_lr=1e-4)
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n_params/1e6:.2f}M params, lr={pg['lr']}")
    
    print("\n✓ Model test passed!")
