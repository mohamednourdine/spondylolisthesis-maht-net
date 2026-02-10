# Component 5: DARK Decoding

## Overview

**DARK (Distribution-Aware Coordinate Representation)** is a post-processing technique for extracting sub-pixel accurate coordinates from heatmaps. Standard argmax only gives integer coordinates, losing precision. DARK recovers sub-pixel accuracy by modeling the heatmap as a 2D Gaussian.

## The Problem with Argmax

```
Standard approach:
    Heatmap [H, W] → argmax → (x_int, y_int) → quantization error

Example:
    True location: (123.7, 456.3)
    After argmax:  (124, 456)
    Error: 0.58 pixels ≈ 0.3 mm (at 0.5 mm/pixel)
```

For clinical use, we need **sub-pixel precision**. A 0.5 pixel error can mean 0.25-0.5 mm error, which accumulates across keypoints.

## DARK Solution

DARK treats the heatmap near the maximum as a 2D Gaussian and uses **Taylor expansion** to find the true peak:

```
Heatmap around peak can be approximated as:
    H(x,y) ≈ H(x₀,y₀) + ∇H·δ + ½δᵀ·(∇²H)·δ

Where:
    (x₀, y₀) = argmax location (integer)
    δ = (Δx, Δy) = offset to true peak
    ∇H = gradient (first derivative)
    ∇²H = Hessian (second derivative)

Setting ∇H(x_true, y_true) = 0 and solving:
    δ = -(∇²H)⁻¹ · ∇H

Final coordinate:
    (x, y) = (x₀ + Δx, y₀ + Δy)
```

## Visual Explanation

```
                    Heatmap around peak
                    
         │
    1.0  │        ╱╲
         │       ╱  ╲
    0.8  │      ╱    ╲
         │     ╱      ╲
    0.6  │    ╱        ╲
         │   ╱          ╲
    0.4  │  ╱            ╲
         │ ╱              ╲
    0.2  │╱                ╲
         │                  
    0.0  └──────┼──────────────
             x₀=124  
                  ↑
              argmax gives 124
              
    But true peak is at 123.7!
    
    DARK uses local gradient to find:
    - Gradient points left (positive slope on right)
    - Hessian tells curvature
    - Solve for offset: Δx = -0.3
    - Final: x = 124 - 0.3 = 123.7 ✓
```

## Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np

def dark_decoding(
    heatmaps: torch.Tensor,
    kernel_size: int = 11
) -> torch.Tensor:
    """
    DARK post-processing for sub-pixel coordinate extraction.
    
    Reference: Zhang et al., "Distribution-Aware Coordinate Representation
    for Human Pose Estimation", CVPR 2020.
    
    Args:
        heatmaps: Predicted heatmaps [B, K, H, W]
        kernel_size: Size of Gaussian smoothing kernel (odd number)
        
    Returns:
        Keypoint coordinates [B, K, 2] with (x, y) in range [0, W-1] and [0, H-1]
    """
    B, K, H, W = heatmaps.shape
    
    # Step 1: Apply Gaussian smoothing to reduce noise
    heatmaps_smooth = gaussian_blur(heatmaps, kernel_size)
    
    # Step 2: Find argmax for each heatmap
    heatmaps_flat = heatmaps_smooth.reshape(B, K, -1)
    max_indices = heatmaps_flat.argmax(dim=-1)  # [B, K]
    
    # Convert to 2D coordinates
    y_int = max_indices // W
    x_int = max_indices % W
    
    # Step 3: Compute sub-pixel offset using Taylor expansion
    offsets = compute_dark_offset(heatmaps_smooth, x_int, y_int)
    
    # Step 4: Add offset to integer coordinates
    x = x_int.float() + offsets[:, :, 0]
    y = y_int.float() + offsets[:, :, 1]
    
    # Clamp to valid range
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)
    
    # Stack coordinates
    coords = torch.stack([x, y], dim=-1)  # [B, K, 2]
    
    return coords


def gaussian_blur(heatmaps: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply Gaussian blur to heatmaps.
    
    Args:
        heatmaps: [B, K, H, W]
        kernel_size: Kernel size (must be odd)
        
    Returns:
        Smoothed heatmaps [B, K, H, W]
    """
    # Create Gaussian kernel
    sigma = (kernel_size - 1) / 6.0
    coords = torch.arange(kernel_size, dtype=torch.float32, device=heatmaps.device)
    coords -= (kernel_size - 1) / 2.0
    
    # 1D Gaussian
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # 2D Gaussian kernel
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    # Expand kernel for all channels
    B, K, H, W = heatmaps.shape
    kernel = kernel.expand(K, 1, -1, -1)
    
    # Apply convolution (groups=K for channel-wise)
    padding = kernel_size // 2
    smoothed = F.conv2d(
        heatmaps.reshape(B * K, 1, H, W),
        kernel[:1],
        padding=padding,
        groups=1
    )
    
    return smoothed.reshape(B, K, H, W)


def compute_dark_offset(
    heatmaps: torch.Tensor,
    x_int: torch.Tensor,
    y_int: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute sub-pixel offset using Taylor expansion.
    
    Uses numerical derivatives at the integer peak location.
    
    Args:
        heatmaps: [B, K, H, W]
        x_int: [B, K] integer x coordinates
        y_int: [B, K] integer y coordinates
        eps: Small value for numerical stability
        
    Returns:
        Offsets [B, K, 2] with (dx, dy)
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    offsets = torch.zeros(B, K, 2, device=device)
    
    for b in range(B):
        for k in range(K):
            x, y = int(x_int[b, k].item()), int(y_int[b, k].item())
            
            # Check bounds (need neighbors for derivatives)
            if x < 1 or x >= W - 1 or y < 1 or y >= H - 1:
                continue
            
            # Get local 3×3 patch
            patch = heatmaps[b, k, y-1:y+2, x-1:x+2]  # [3, 3]
            
            # Compute derivatives (central differences)
            # First derivatives
            dx = (patch[1, 2] - patch[1, 0]) / 2.0  # dH/dx
            dy = (patch[2, 1] - patch[0, 1]) / 2.0  # dH/dy
            
            # Second derivatives (Hessian)
            dxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]  # d²H/dx²
            dyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]  # d²H/dy²
            dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0  # d²H/dxdy
            
            # Hessian matrix
            # H = [[dxx, dxy],
            #      [dxy, dyy]]
            
            # Determinant
            det = dxx * dyy - dxy * dxy
            
            # Check for valid Hessian (should be negative definite at maximum)
            if abs(det) < eps or dxx >= 0:
                continue
            
            # Solve for offset: δ = -H⁻¹ · ∇H
            # H⁻¹ = (1/det) * [[dyy, -dxy], [-dxy, dxx]]
            offset_x = -(dyy * dx - dxy * dy) / det
            offset_y = -(dxx * dy - dxy * dx) / det
            
            # Clamp offset to reasonable range (within ±0.5 pixels)
            offset_x = max(-0.5, min(0.5, offset_x.item()))
            offset_y = max(-0.5, min(0.5, offset_y.item()))
            
            offsets[b, k, 0] = offset_x
            offsets[b, k, 1] = offset_y
    
    return offsets


def dark_decoding_vectorized(
    heatmaps: torch.Tensor,
    kernel_size: int = 11
) -> torch.Tensor:
    """
    Vectorized version of DARK decoding (faster for GPU).
    
    Args:
        heatmaps: [B, K, H, W]
        kernel_size: Gaussian blur kernel size
        
    Returns:
        Coordinates [B, K, 2]
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    # Gaussian smoothing
    heatmaps_smooth = gaussian_blur(heatmaps, kernel_size)
    
    # Find argmax
    heatmaps_flat = heatmaps_smooth.reshape(B, K, -1)
    max_vals, max_indices = heatmaps_flat.max(dim=-1)
    
    y_int = (max_indices // W).float()
    x_int = (max_indices % W).float()
    
    # Compute derivatives using convolution
    # Sobel-like kernels for derivatives
    dx_kernel = torch.tensor([[[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]]], dtype=torch.float32, device=device) / 8.0
    
    dy_kernel = torch.tensor([[[[-1, -2, -1],
                                 [0,  0,  0],
                                 [1,  2,  1]]]], dtype=torch.float32, device=device) / 8.0
    
    # Laplacian for second derivatives
    dxx_kernel = torch.tensor([[[[0, 0, 0],
                                  [1, -2, 1],
                                  [0, 0, 0]]]], dtype=torch.float32, device=device)
    
    dyy_kernel = torch.tensor([[[[0, 1, 0],
                                  [0, -2, 0],
                                  [0, 1, 0]]]], dtype=torch.float32, device=device)
    
    # Apply derivative kernels
    hm_pad = F.pad(heatmaps_smooth, (1, 1, 1, 1), mode='replicate')
    hm_flat = hm_pad.reshape(B * K, 1, H + 2, W + 2)
    
    dx_map = F.conv2d(hm_flat, dx_kernel).reshape(B, K, H, W)
    dy_map = F.conv2d(hm_flat, dy_kernel).reshape(B, K, H, W)
    dxx_map = F.conv2d(hm_flat, dxx_kernel).reshape(B, K, H, W)
    dyy_map = F.conv2d(hm_flat, dyy_kernel).reshape(B, K, H, W)
    
    # Sample derivatives at peak locations
    # Create grid for sampling
    grid_y = (y_int / (H - 1) * 2 - 1).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
    grid_x = (x_int / (W - 1) * 2 - 1).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
    grid = torch.cat([grid_x, grid_y], dim=-1)  # [B, K, 1, 2]
    
    # Sample
    dx = F.grid_sample(dx_map, grid, align_corners=True).squeeze(-1).squeeze(-1)
    dy = F.grid_sample(dy_map, grid, align_corners=True).squeeze(-1).squeeze(-1)
    dxx = F.grid_sample(dxx_map, grid, align_corners=True).squeeze(-1).squeeze(-1)
    dyy = F.grid_sample(dyy_map, grid, align_corners=True).squeeze(-1).squeeze(-1)
    
    # Compute offsets
    det = dxx * dyy
    valid = (det.abs() > 1e-6) & (dxx < 0)
    
    offset_x = torch.zeros_like(dx)
    offset_y = torch.zeros_like(dy)
    
    offset_x[valid] = -dx[valid] / dxx[valid]
    offset_y[valid] = -dy[valid] / dyy[valid]
    
    # Clamp
    offset_x = offset_x.clamp(-0.5, 0.5)
    offset_y = offset_y.clamp(-0.5, 0.5)
    
    # Final coordinates
    x = x_int + offset_x
    y = y_int + offset_y
    
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)
    
    return torch.stack([x, y], dim=-1)


# Alternative: Soft-argmax (differentiable)
def soft_argmax(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable coordinate extraction using soft-argmax.
    
    Unlike DARK (post-processing), soft-argmax is differentiable
    and can be used during training.
    
    Args:
        heatmaps: [B, K, H, W]
        temperature: Softmax temperature (lower = sharper)
        
    Returns:
        Coordinates [B, K, 2]
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    
    # Apply softmax to get weights
    heatmaps_flat = heatmaps.reshape(B, K, -1)
    weights = F.softmax(heatmaps_flat / temperature, dim=-1)
    weights = weights.reshape(B, K, H, W)
    
    # Compute expected coordinates
    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  # [B, K]
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  # [B, K]
    
    return torch.stack([x, y], dim=-1)


# Confidence extraction
def extract_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract confidence scores from heatmap peaks.
    
    Args:
        heatmaps: [B, K, H, W]
        
    Returns:
        Confidence scores [B, K]
    """
    # Max value as confidence
    confidence = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], -1).max(dim=-1)[0]
    return confidence


# Usage example
if __name__ == "__main__":
    # Simulate predicted heatmaps
    batch_size = 4
    num_keypoints = 20
    heatmaps = torch.randn(batch_size, num_keypoints, 512, 512)
    
    # Add Gaussian peaks at known locations
    for k in range(num_keypoints):
        x_true, y_true = 100.3 + k * 20, 150.7 + k * 15
        y_grid, x_grid = torch.meshgrid(
            torch.arange(512, dtype=torch.float32),
            torch.arange(512, dtype=torch.float32),
            indexing='ij'
        )
        gaussian = torch.exp(-((x_grid - x_true)**2 + (y_grid - y_true)**2) / (2 * 2**2))
        heatmaps[0, k] += gaussian * 5  # Add strong peak
    
    # Extract coordinates
    coords_dark = dark_decoding(heatmaps[:1])
    coords_soft = soft_argmax(heatmaps[:1])
    
    print(f"DARK coordinates for keypoint 0: {coords_dark[0, 0]}")
    print(f"Soft-argmax coordinates for keypoint 0: {coords_soft[0, 0]}")
    print(f"True location: (100.3, 150.7)")
```

## Comparison: Argmax vs Soft-Argmax vs DARK

| Method | Sub-pixel | Differentiable | Pros | Cons |
|--------|-----------|----------------|------|------|
| **Argmax** | No | No | Simple, fast | Integer only |
| **Soft-argmax** | Yes | Yes | Can be used in loss | Sensitive to background |
| **DARK** | Yes | No (post-proc) | Most accurate | Post-processing only |

## Key Design Decisions

### 1. Gaussian Smoothing

**Decision**: Apply Gaussian blur before DARK.

**Rationale**:
- Reduces noise in predicted heatmaps
- Makes derivative computation more stable
- σ = (kernel_size - 1) / 6 is a good default

### 2. Offset Clamping

**Decision**: Clamp offsets to ±0.5 pixels.

**Rationale**:
- If offset > 0.5, the integer argmax was wrong
- Large offsets indicate unreliable predictions
- Clamping prevents catastrophic errors

### 3. When to Use DARK vs Soft-Argmax

| Scenario | Recommendation |
|----------|----------------|
| **Training** | Soft-argmax (differentiable) |
| **Validation/Test** | DARK (most accurate) |
| **End-to-end training** | Heatmap loss (no coordinate extraction needed) |

## Performance Impact

| Decoding Method | Error Reduction | Time Overhead |
|-----------------|-----------------|---------------|
| Argmax only | Baseline | ~1ms |
| Soft-argmax | ~10-15% better | ~5ms |
| DARK | ~20-25% better | ~10ms |

DARK typically reduces localization error by 0.1-0.3 pixels, which translates to ~0.05-0.15 mm at typical pixel spacings.

## References

1. Zhang, F., et al. (2020). Distribution-Aware Coordinate Representation for Human Pose Estimation. CVPR.
2. Sun, X., et al. (2018). Integral Human Pose Regression. ECCV. (Soft-argmax)
3. Nibali, A., et al. (2018). Numerical Coordinate Regression with Convolutional Neural Networks. (DSNT)
