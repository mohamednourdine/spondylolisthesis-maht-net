"""
UNet model for keypoint detection via heatmap regression.
Predicts heatmaps for vertebra corner keypoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input sizes that are not divisible by 16
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for keypoint heatmap prediction.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_keypoints: Number of keypoint types (4 for vertebra corners)
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False)
        base_channels: Number of channels in first layer (default: 64)
    """
    
    def __init__(self, in_channels=3, num_keypoints=4, bilinear=False, base_channels=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer - one heatmap per keypoint type
        self.outc = OutConv(base_channels, num_keypoints)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output heatmaps
        logits = self.outc(x)
        return logits
    
    def predict_keypoints(self, x, threshold=0.5):
        """
        Predict keypoint locations from input image.
        
        Args:
            x: Input image tensor [B, C, H, W]
            threshold: Confidence threshold for keypoint detection
            
        Returns:
            List of keypoint arrays for each image in batch
        """
        self.eval()
        with torch.no_grad():
            heatmaps = torch.sigmoid(self.forward(x))
        
        batch_keypoints = []
        for b in range(heatmaps.shape[0]):
            keypoints = []
            for k in range(self.num_keypoints):
                heatmap = heatmaps[b, k].cpu().numpy()
                
                # Find peaks in heatmap
                from scipy.ndimage import maximum_filter
                local_max = maximum_filter(heatmap, size=3) == heatmap
                peaks = (heatmap > threshold) & local_max
                
                # Get coordinates
                y_coords, x_coords = np.where(peaks)
                confidences = heatmap[peaks]
                
                # Store as [x, y, confidence]
                kp_list = np.stack([x_coords, y_coords, confidences], axis=1)
                keypoints.append(kp_list)
            
            batch_keypoints.append(keypoints)
        
        return batch_keypoints


def create_unet(in_channels=3, num_keypoints=4, pretrained=False, **kwargs):
    """
    Factory function to create UNet model.
    
    Args:
        in_channels: Number of input channels
        num_keypoints: Number of keypoint types
        pretrained: Whether to load pretrained weights (not implemented yet)
        **kwargs: Additional arguments for UNet
        
    Returns:
        UNet model
    """
    model = UNet(in_channels=in_channels, num_keypoints=num_keypoints, **kwargs)
    
    if pretrained:
        # TODO: Load pretrained weights if available
        print("Warning: Pretrained weights not yet implemented for UNet")
    
    return model