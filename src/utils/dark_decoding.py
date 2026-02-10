"""
DARK Decoding: Sub-pixel coordinate extraction from heatmaps.

DARK (Distribution-Aware Coordinate Representation) uses Taylor expansion
to find sub-pixel accurate coordinates from heatmap peaks.

Reference: Zhang et al., "Distribution-Aware Coordinate Representation
for Human Pose Estimation", CVPR 2020.

This module provides:
- dark_decoding: Full DARK implementation with Gaussian smoothing
- soft_argmax: Differentiable alternative for training
- extract_confidence: Confidence scores from heatmap peaks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def gaussian_blur(heatmaps: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    """
    Apply Gaussian blur to heatmaps for noise reduction.
    
    Args:
        heatmaps: (B, K, H, W) input heatmaps
        kernel_size: Size of Gaussian kernel (must be odd)
        
    Returns:
        Smoothed heatmaps (B, K, H, W)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype
    
    # Create 1D Gaussian kernel
    sigma = (kernel_size - 1) / 6.0
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D Gaussian kernel
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # (K, K)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    
    # Apply channel-wise convolution
    padding = kernel_size // 2
    heatmaps_reshaped = heatmaps.view(B * K, 1, H, W)
    smoothed = F.conv2d(heatmaps_reshaped, kernel, padding=padding)
    smoothed = smoothed.view(B, K, H, W)
    
    return smoothed


def dark_decoding(
    heatmaps: torch.Tensor,
    kernel_size: int = 11,
    apply_smoothing: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DARK decoding for sub-pixel coordinate extraction.
    
    Uses Taylor expansion around the heatmap peak to find
    sub-pixel accurate coordinates.
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        kernel_size: Gaussian smoothing kernel size
        apply_smoothing: Whether to apply Gaussian blur first
        
    Returns:
        coords: (B, K, 2) refined coordinates (x, y)
        confidence: (B, K) confidence scores (peak values)
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    # Step 1: Gaussian smoothing for noise reduction
    if apply_smoothing:
        heatmaps_smooth = gaussian_blur(heatmaps, kernel_size)
    else:
        heatmaps_smooth = heatmaps
    
    # Step 2: Find argmax locations
    heatmaps_flat = heatmaps_smooth.view(B, K, -1)
    max_vals, max_indices = heatmaps_flat.max(dim=-1)
    
    y_int = (max_indices // W).float()
    x_int = (max_indices % W).float()
    
    # Step 3: Compute sub-pixel offsets using Taylor expansion
    offsets = _compute_dark_offsets(heatmaps_smooth, x_int.long(), y_int.long())
    
    # Step 4: Add offsets to integer coordinates
    x = x_int + offsets[..., 0]
    y = y_int + offsets[..., 1]
    
    # Clamp to valid range
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)
    
    coords = torch.stack([x, y], dim=-1)  # (B, K, 2)
    
    return coords, max_vals


def _compute_dark_offsets(
    heatmaps: torch.Tensor,
    x_int: torch.Tensor,
    y_int: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute sub-pixel offsets using Taylor expansion.
    
    At the peak location, we approximate the heatmap as a 2D quadratic:
        H(x,y) ≈ H(x₀,y₀) + ∇H·δ + ½δᵀ·Hessian·δ
    
    Setting gradient to zero and solving gives:
        δ = -Hessian⁻¹ · ∇H
    
    Args:
        heatmaps: (B, K, H, W) smoothed heatmaps
        x_int: (B, K) integer x coordinates
        y_int: (B, K) integer y coordinates
        eps: Small value for numerical stability
        
    Returns:
        offsets: (B, K, 2) sub-pixel offsets (dx, dy)
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    offsets = torch.zeros(B, K, 2, device=device, dtype=heatmaps.dtype)
    
    for b in range(B):
        for k in range(K):
            x, y = x_int[b, k].item(), y_int[b, k].item()
            
            # Check bounds (need 3x3 neighborhood for derivatives)
            if x < 1 or x >= W - 1 or y < 1 or y >= H - 1:
                continue
            
            # Extract 3x3 patch around peak
            patch = heatmaps[b, k, y-1:y+2, x-1:x+2]  # (3, 3)
            
            # Compute first derivatives (central differences)
            dx = (patch[1, 2] - patch[1, 0]) / 2.0  # dH/dx
            dy = (patch[2, 1] - patch[0, 1]) / 2.0  # dH/dy
            
            # Compute second derivatives (Hessian)
            dxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]  # d²H/dx²
            dyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]  # d²H/dy²
            dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0  # d²H/dxdy
            
            # Compute determinant of Hessian
            det = dxx * dyy - dxy * dxy
            
            # Check for valid maximum (Hessian should be negative definite)
            if abs(det) < eps or dxx >= 0:
                continue
            
            # Solve: δ = -H⁻¹ · ∇H
            # H⁻¹ = (1/det) * [[dyy, -dxy], [-dxy, dxx]]
            offset_x = -(dyy * dx - dxy * dy) / det
            offset_y = -(dxx * dy - dxy * dx) / det
            
            # Clamp offsets to ±0.5 (if larger, argmax was likely wrong)
            offset_x = max(-0.5, min(0.5, offset_x.item()))
            offset_y = max(-0.5, min(0.5, offset_y.item()))
            
            offsets[b, k, 0] = offset_x
            offsets[b, k, 1] = offset_y
    
    return offsets


def soft_argmax(
    heatmaps: torch.Tensor,
    temperature: float = 1.0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Differentiable coordinate extraction using soft-argmax.
    
    Unlike DARK, soft-argmax is differentiable and can be used
    during training for direct coordinate supervision.
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        temperature: Softmax temperature (lower = sharper peaks)
        normalize: Whether to normalize heatmaps before softmax
        
    Returns:
        coords: (B, K, 2) coordinates (x, y)
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype
    
    # Create coordinate grids
    y_coords = torch.arange(H, dtype=dtype, device=device)
    x_coords = torch.arange(W, dtype=dtype, device=device)
    
    # Flatten heatmaps and apply softmax
    heatmaps_flat = heatmaps.view(B, K, -1)
    
    if normalize:
        # Normalize to [0, 1] range
        hm_min = heatmaps_flat.min(dim=-1, keepdim=True)[0]
        hm_max = heatmaps_flat.max(dim=-1, keepdim=True)[0]
        heatmaps_flat = (heatmaps_flat - hm_min) / (hm_max - hm_min + 1e-8)
    
    weights = F.softmax(heatmaps_flat / temperature, dim=-1)
    weights = weights.view(B, K, H, W)
    
    # Compute expected coordinates
    # x = sum_y(sum_x(weight * x)) = sum_x(sum_y(weight) * x)
    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  # (B, K)
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  # (B, K)
    
    return torch.stack([x, y], dim=-1)


def extract_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract confidence scores from heatmap peaks.
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        
    Returns:
        confidence: (B, K) confidence scores
    """
    B, K, H, W = heatmaps.shape
    confidence = heatmaps.view(B, K, -1).max(dim=-1)[0]
    return confidence


def extract_heatmap_uncertainty(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract uncertainty from heatmap spread.
    
    Measures the spatial spread of the heatmap around its peak
    as an indicator of prediction uncertainty.
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        
    Returns:
        sigma_x: (B, K) uncertainty in x direction
        sigma_y: (B, K) uncertainty in y direction
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype
    
    # Create coordinate grids
    y_coords = torch.arange(H, dtype=dtype, device=device)
    x_coords = torch.arange(W, dtype=dtype, device=device)
    
    # Convert heatmaps to probability distributions
    heatmaps_flat = heatmaps.view(B, K, -1)
    probs = F.softmax(heatmaps_flat, dim=-1)
    probs = probs.view(B, K, H, W)
    
    # Compute expected coordinates (mean)
    x_mean = (probs.sum(dim=2) * x_coords).sum(dim=-1)  # (B, K)
    y_mean = (probs.sum(dim=3) * y_coords).sum(dim=-1)  # (B, K)
    
    # Compute variance
    x_var = (probs.sum(dim=2) * (x_coords - x_mean.unsqueeze(-1))**2).sum(dim=-1)
    y_var = (probs.sum(dim=3) * (y_coords - y_mean.unsqueeze(-1))**2).sum(dim=-1)
    
    # Standard deviation
    sigma_x = torch.sqrt(x_var + 1e-8)
    sigma_y = torch.sqrt(y_var + 1e-8)
    
    return sigma_x, sigma_y


class DARKDecoding(nn.Module):
    """
    DARK decoding as a module for easy integration.
    
    Can be used as a post-processing step after heatmap prediction.
    """
    
    def __init__(
        self,
        kernel_size: int = 11,
        apply_smoothing: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.apply_smoothing = apply_smoothing
    
    def forward(
        self,
        heatmaps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract coordinates using DARK decoding.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            
        Returns:
            coords: (B, K, 2) refined coordinates
            confidence: (B, K) confidence scores
        """
        return dark_decoding(
            heatmaps,
            kernel_size=self.kernel_size,
            apply_smoothing=self.apply_smoothing
        )


class SoftArgmax(nn.Module):
    """
    Soft-argmax as a differentiable module for training.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract coordinates using soft-argmax.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            
        Returns:
            coords: (B, K, 2) coordinates
        """
        return soft_argmax(heatmaps, self.temperature)


def test_dark_decoding():
    """Test DARK decoding implementation."""
    print("\n" + "="*60)
    print("Testing DARK Decoding")
    print("="*60)
    
    B, K, H, W = 2, 20, 512, 512
    
    # Create test heatmaps with known peak locations
    heatmaps = torch.zeros(B, K, H, W)
    true_coords = torch.zeros(B, K, 2)
    
    for b in range(B):
        for k in range(K):
            # True sub-pixel location
            x_true = 100.3 + k * 20
            y_true = 150.7 + k * 15
            
            if x_true < W and y_true < H:
                true_coords[b, k] = torch.tensor([x_true, y_true])
                
                # Create Gaussian heatmap
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, dtype=torch.float32),
                    torch.arange(W, dtype=torch.float32),
                    indexing='ij'
                )
                sigma = 2.0
                gaussian = torch.exp(-((x_grid - x_true)**2 + (y_grid - y_true)**2) / (2 * sigma**2))
                heatmaps[b, k] = gaussian
    
    # Test DARK decoding
    coords_dark, confidence = dark_decoding(heatmaps)
    
    # Test soft-argmax
    coords_soft = soft_argmax(heatmaps)
    
    # Compute errors
    dark_error = torch.sqrt(((coords_dark - true_coords)**2).sum(dim=-1)).mean()
    soft_error = torch.sqrt(((coords_soft - true_coords)**2).sum(dim=-1)).mean()
    
    print(f"\nTrue location (kp 0): ({true_coords[0, 0, 0]:.2f}, {true_coords[0, 0, 1]:.2f})")
    print(f"DARK result:  ({coords_dark[0, 0, 0]:.2f}, {coords_dark[0, 0, 1]:.2f})")
    print(f"Soft-argmax:  ({coords_soft[0, 0, 0]:.2f}, {coords_soft[0, 0, 1]:.2f})")
    print(f"\nMean DARK error: {dark_error:.4f} pixels")
    print(f"Mean soft-argmax error: {soft_error:.4f} pixels")
    print(f"Confidence sample: {confidence[0, 0]:.4f}")
    
    # Test uncertainty extraction
    sigma_x, sigma_y = extract_heatmap_uncertainty(heatmaps)
    print(f"\nUncertainty sample: σ_x={sigma_x[0, 0]:.2f}, σ_y={sigma_y[0, 0]:.2f}")
    
    assert dark_error < 0.5, f"DARK error too high: {dark_error}"
    print("\n✓ DARK decoding test passed!")
    return True


if __name__ == "__main__":
    test_dark_decoding()
