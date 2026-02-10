# Component 6: Loss Functions

## Overview

MAHT-Net uses a **multi-component loss** combining different objectives:

```python
L_total = λ₁·L_heatmap + λ₂·L_coordinate + λ₃·L_anatomical + λ₄·L_uncertainty
```

Each component addresses different aspects of keypoint localization.

## Loss Architecture

```
                    Ground Truth
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    GT Heatmaps    GT Coordinates   GT Anatomy
          │              │              │
          │              │              │
    ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
    │ Predicted │  │ Predicted │  │ Predicted │
    │ Heatmaps  │  │   Coords  │  │  Anatomy  │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │
          ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ L_heatmap │  │ L_coord   │  │ L_anatomy │
    │ (MSE/Focal│  │ (Wing)    │  │ (Struct)  │
    └───────────┘  └───────────┘  └───────────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                   L_total (weighted sum)
```

---

## 1. Heatmap Loss (Primary)

### 1.1 MSE Loss (Simple)

```python
L_heatmap = MSE(pred_heatmaps, gt_heatmaps)
         = (1/N) Σ (H_pred - H_gt)²
```

**Pros**: Simple, stable, effective  
**Cons**: Treats all pixels equally (background vs keypoint)

### 1.2 Focal MSE Loss (Class Imbalance)

Most pixels are background (value ≈ 0). Focal loss downweights easy (background) pixels:

```python
L_heatmap = focal_mse(pred, gt) = (1 - gt)^γ · (pred - gt)²
```

Where γ (gamma) controls focus:
- γ = 0: Standard MSE
- γ = 2: Focus on hard examples (keypoint regions)

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapMSELoss(nn.Module):
    """
    Standard MSE loss for heatmaps.
    """
    
    def __init__(self, use_target_weight: bool = False):
        super().__init__()
        self.use_target_weight = use_target_weight
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps [B, K, H, W]
            target: Ground truth heatmaps [B, K, H, W]
            target_weight: Per-keypoint weights [B, K] (optional)
            
        Returns:
            Scalar loss
        """
        batch_size = pred.size(0)
        num_keypoints = pred.size(1)
        
        # Flatten spatial dimensions
        pred_flat = pred.reshape(batch_size, num_keypoints, -1)
        target_flat = target.reshape(batch_size, num_keypoints, -1)
        
        # MSE per keypoint
        loss = ((pred_flat - target_flat) ** 2).mean(dim=-1)  # [B, K]
        
        # Apply target weight if provided
        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight
        
        return loss.mean()


class FocalHeatmapLoss(nn.Module):
    """
    Focal MSE loss for heatmaps.
    
    Down-weights easy background pixels to focus on keypoint regions.
    """
    
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, K, H, W]
            target: [B, K, H, W]
        """
        # Focal weight: (1 - target)^gamma
        # High target (keypoint) → low weight reduction
        # Low target (background) → high weight reduction
        focal_weight = (1 - target) ** self.gamma
        
        # Weighted MSE
        loss = focal_weight * ((pred - target) ** 2)
        
        return loss.mean()


class AWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression.
    
    Combines benefits of L1 (robust) and L2 (smooth gradient) losses,
    with adaptive weighting for foreground pixels.
    
    Reference: Wang et al., "Adaptive Wing Loss for Robust Face Alignment
    via Heatmap Regression", ICCV 2019.
    """
    
    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, K, H, W]
            target: [B, K, H, W]
        """
        delta = (pred - target).abs()
        
        # Different loss for small and large errors
        # Small errors: modified log loss (smooth gradient)
        # Large errors: linear (robust to outliers)
        
        A = self.omega * (
            1 / (1 + (self.theta / self.epsilon) ** (self.alpha - target))
        ) * (self.alpha - target) * (
            (self.theta / self.epsilon) ** (self.alpha - target - 1)
        ) / self.epsilon
        
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - target)
        )
        
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(
                1 + (delta / self.epsilon) ** (self.alpha - target)
            ),
            A * delta - C
        )
        
        return losses.mean()
```

---

## 2. Coordinate Loss (Refinement)

Direct supervision on extracted coordinates provides gradient even when heatmap is correct but imprecise.

### 2.1 Wing Loss

Better than L1/L2 for keypoint localization:

```
Wing(x) = ω · ln(1 + |x|/ε)     if |x| < ω
        = |x| - C               otherwise

Where C = ω - ω·ln(1 + ω/ε)
```

**Key properties**:
- Small errors (< ω): Log-like, large gradient → precise refinement
- Large errors (≥ ω): Linear → robust to outliers

### Implementation

```python
class WingLoss(nn.Module):
    """
    Wing Loss for coordinate regression.
    
    Reference: Wu et al., "Look at Boundary: A Boundary-Aware Face
    Alignment Algorithm", CVPR 2018.
    """
    
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        """
        Args:
            omega: Threshold between log and linear regions
            epsilon: Curvature of log region
        """
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted coordinates [B, K, 2]
            target: Ground truth coordinates [B, K, 2]
            
        Returns:
            Scalar loss
        """
        delta = (pred - target).abs()
        
        # Precompute constant
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        
        # Compute loss
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - C
        )
        
        return losses.mean()


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss (Huber loss) - simpler alternative to Wing.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta)
```

---

## 3. Anatomical Structure Loss (Novel)

**Our contribution**: Penalize anatomically impossible predictions.

### 3.1 Loss Components

```python
L_anatomical = L_order + λ₁·L_parallel + λ₂·L_ratio
```

1. **L_order**: Vertebrae must be ordered top-to-bottom
2. **L_parallel**: Vertebral edges should be roughly parallel
3. **L_ratio**: Vertebral proportions should be reasonable

### Implementation

```python
class AnatomicalStructureLoss(nn.Module):
    """
    Anatomical structure loss for vertebral keypoints.
    
    Penalizes predictions that violate known anatomical constraints:
    - Vertebrae must be vertically ordered (L1 above L2, etc.)
    - Vertebral bodies should be roughly rectangular
    - Adjacent vertebrae should be similar in size
    """
    
    def __init__(
        self,
        lambda_order: float = 1.0,
        lambda_parallel: float = 0.1,
        lambda_ratio: float = 0.1,
        view: str = 'AP'
    ):
        super().__init__()
        self.lambda_order = lambda_order
        self.lambda_parallel = lambda_parallel
        self.lambda_ratio = lambda_ratio
        self.view = view
        
    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints: Predicted keypoints [B, K, 2]
                K = 20 for AP (5 vertebrae × 4 corners)
                K = 22 for LA (5 vertebrae × 4 corners + 2 S1)
                
        Returns:
            Scalar anatomical loss
        """
        losses = []
        
        # 1. Ordering loss
        L_order = self._compute_ordering_loss(keypoints)
        losses.append(self.lambda_order * L_order)
        
        # 2. Parallelism loss
        L_parallel = self._compute_parallel_loss(keypoints)
        losses.append(self.lambda_parallel * L_parallel)
        
        # 3. Ratio loss
        L_ratio = self._compute_ratio_loss(keypoints)
        losses.append(self.lambda_ratio * L_ratio)
        
        return sum(losses)
    
    def _compute_ordering_loss(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Ensure vertebrae are ordered top-to-bottom (increasing y).
        
        L1_y < L2_y < L3_y < L4_y < L5_y
        """
        B, K, _ = keypoints.shape
        
        # Get vertebra centers (average of 4 corners)
        num_vertebrae = 5 if self.view == 'AP' else 6
        centers = []
        
        for v in range(num_vertebrae):
            start_idx = v * 4
            end_idx = start_idx + 4 if v < 5 else start_idx + 2  # S1 only has 2 points
            vertebra_points = keypoints[:, start_idx:end_idx, :]
            center = vertebra_points.mean(dim=1)  # [B, 2]
            centers.append(center)
        
        centers = torch.stack(centers, dim=1)  # [B, V, 2]
        
        # Compute ordering violations
        # y-coordinate should increase going down (L1 → L5)
        y_coords = centers[:, :, 1]  # [B, V]
        
        # Difference between consecutive vertebrae
        y_diff = y_coords[:, 1:] - y_coords[:, :-1]  # [B, V-1]
        
        # Penalize negative differences (wrong order)
        violations = F.relu(-y_diff)  # Only penalize if next vertebra is above
        
        return violations.mean()
    
    def _compute_parallel_loss(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Ensure upper and lower edges of each vertebra are roughly parallel.
        
        For each vertebra:
        - Upper edge: UL → UR
        - Lower edge: LL → LR
        These should have similar slopes.
        """
        B = keypoints.shape[0]
        losses = []
        
        for v in range(5):  # L1-L5
            # Corner indices for vertebra v
            # Layout: [UL, UR, LL, LR]
            base = v * 4
            ul, ur, ll, lr = base, base + 1, base + 2, base + 3
            
            # Upper edge vector
            upper_vec = keypoints[:, ur, :] - keypoints[:, ul, :]  # [B, 2]
            
            # Lower edge vector
            lower_vec = keypoints[:, lr, :] - keypoints[:, ll, :]  # [B, 2]
            
            # Compute angle difference using dot product
            # If parallel, angle between them is 0 or 180 degrees
            upper_norm = upper_vec / (upper_vec.norm(dim=-1, keepdim=True) + 1e-6)
            lower_norm = lower_vec / (lower_vec.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Dot product should be close to ±1
            dot = (upper_norm * lower_norm).sum(dim=-1).abs()  # [B]
            
            # Loss: 1 - |dot product|
            losses.append(1 - dot)
        
        return torch.stack(losses, dim=1).mean()
    
    def _compute_ratio_loss(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Ensure vertebral proportions are reasonable.
        
        Width/height ratio should be within expected range.
        Adjacent vertebrae should have similar sizes.
        """
        B = keypoints.shape[0]
        widths = []
        heights = []
        
        for v in range(5):  # L1-L5
            base = v * 4
            ul, ur, ll, lr = base, base + 1, base + 2, base + 3
            
            # Width: average of upper and lower edge lengths
            upper_width = (keypoints[:, ur, :] - keypoints[:, ul, :]).norm(dim=-1)
            lower_width = (keypoints[:, lr, :] - keypoints[:, ll, :]).norm(dim=-1)
            width = (upper_width + lower_width) / 2  # [B]
            
            # Height: average of left and right edge lengths
            left_height = (keypoints[:, ll, :] - keypoints[:, ul, :]).norm(dim=-1)
            right_height = (keypoints[:, lr, :] - keypoints[:, ur, :]).norm(dim=-1)
            height = (left_height + right_height) / 2  # [B]
            
            widths.append(width)
            heights.append(height)
        
        widths = torch.stack(widths, dim=1)   # [B, 5]
        heights = torch.stack(heights, dim=1)  # [B, 5]
        
        # 1. Aspect ratio loss (width/height should be reasonable)
        ratios = widths / (heights + 1e-6)
        # Typical lumbar vertebra: width ≈ 1.5-3× height
        ratio_loss = F.relu(ratios - 4.0) + F.relu(0.5 - ratios)
        
        # 2. Size consistency loss (adjacent vertebrae similar in size)
        width_diff = (widths[:, 1:] - widths[:, :-1]).abs()
        height_diff = (heights[:, 1:] - heights[:, :-1]).abs()
        
        # Normalize by average size
        avg_width = widths.mean(dim=1, keepdim=True)
        avg_height = heights.mean(dim=1, keepdim=True)
        
        consistency_loss = (
            (width_diff / (avg_width + 1e-6)).mean() +
            (height_diff / (avg_height + 1e-6)).mean()
        )
        
        return ratio_loss.mean() + 0.5 * consistency_loss
```

---

## 4. Uncertainty Loss (Optional)

For clinical reliability, we can estimate and calibrate uncertainty.

```python
class UncertaintyLoss(nn.Module):
    """
    Negative log-likelihood loss with learned uncertainty.
    
    Model predicts both coordinates and uncertainty (variance).
    Loss: NLL = (pred - target)² / (2σ²) + log(σ)
    
    This encourages:
    - High uncertainty when errors are large (honest about mistakes)
    - Low uncertainty when predictions are accurate
    """
    
    def __init__(self, min_sigma: float = 0.1, max_sigma: float = 10.0):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted coordinates [B, K, 2]
            target: Ground truth [B, K, 2]
            log_sigma: Predicted log-variance [B, K] or [B, K, 2]
            
        Returns:
            Calibrated NLL loss
        """
        # Clamp sigma for stability
        sigma = torch.exp(log_sigma).clamp(self.min_sigma, self.max_sigma)
        
        # Squared error
        sq_error = (pred - target) ** 2
        
        # If sigma is [B, K], expand to [B, K, 2]
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(-1).expand_as(sq_error)
        
        # NLL
        nll = sq_error / (2 * sigma ** 2) + torch.log(sigma)
        
        return nll.mean()
```

---

## 5. Combined Loss

```python
class MAHTNetLoss(nn.Module):
    """
    Combined loss for MAHT-Net training.
    
    L_total = λ₁·L_heatmap + λ₂·L_coord + λ₃·L_anatomical + λ₄·L_uncertainty
    """
    
    def __init__(
        self,
        lambda_heatmap: float = 1.0,
        lambda_coord: float = 0.5,
        lambda_anatomical: float = 0.1,
        lambda_uncertainty: float = 0.1,
        use_focal: bool = True,
        view: str = 'AP'
    ):
        super().__init__()
        
        self.lambda_heatmap = lambda_heatmap
        self.lambda_coord = lambda_coord
        self.lambda_anatomical = lambda_anatomical
        self.lambda_uncertainty = lambda_uncertainty
        
        # Loss components
        if use_focal:
            self.heatmap_loss = FocalHeatmapLoss(gamma=2.0)
        else:
            self.heatmap_loss = HeatmapMSELoss()
        
        self.coord_loss = WingLoss(omega=10.0, epsilon=2.0)
        self.anatomical_loss = AnatomicalStructureLoss(view=view)
        self.uncertainty_loss = UncertaintyLoss()
        
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        pred_coords: torch.Tensor = None,
        gt_coords: torch.Tensor = None,
        pred_log_sigma: torch.Tensor = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred_heatmaps: [B, K, H, W]
            gt_heatmaps: [B, K, H, W]
            pred_coords: [B, K, 2] (optional)
            gt_coords: [B, K, 2] (optional)
            pred_log_sigma: [B, K] (optional)
            
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # 1. Heatmap loss (always used)
        L_heatmap = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
        losses['heatmap'] = L_heatmap
        total = self.lambda_heatmap * L_heatmap
        
        # 2. Coordinate loss (if coordinates provided)
        if pred_coords is not None and gt_coords is not None:
            L_coord = self.coord_loss(pred_coords, gt_coords)
            losses['coord'] = L_coord
            total = total + self.lambda_coord * L_coord
        
        # 3. Anatomical loss (if coordinates provided)
        if pred_coords is not None:
            L_anatomical = self.anatomical_loss(pred_coords)
            losses['anatomical'] = L_anatomical
            total = total + self.lambda_anatomical * L_anatomical
        
        # 4. Uncertainty loss (if uncertainty provided)
        if pred_log_sigma is not None and pred_coords is not None and gt_coords is not None:
            L_uncertainty = self.uncertainty_loss(pred_coords, gt_coords, pred_log_sigma)
            losses['uncertainty'] = L_uncertainty
            total = total + self.lambda_uncertainty * L_uncertainty
        
        losses['total'] = total
        
        return losses


# Usage example
if __name__ == "__main__":
    # Create loss function
    criterion = MAHTNetLoss(
        lambda_heatmap=1.0,
        lambda_coord=0.5,
        lambda_anatomical=0.1,
        view='AP'
    )
    
    # Simulate predictions and targets
    batch_size = 4
    pred_heatmaps = torch.randn(batch_size, 20, 512, 512)
    gt_heatmaps = torch.randn(batch_size, 20, 512, 512)
    pred_coords = torch.randn(batch_size, 20, 2) * 100 + 256
    gt_coords = torch.randn(batch_size, 20, 2) * 100 + 256
    
    # Compute loss
    losses = criterion(pred_heatmaps, gt_heatmaps, pred_coords, gt_coords)
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
```

## Recommended Loss Weights

| Component | Weight (λ) | Rationale |
|-----------|------------|-----------|
| **L_heatmap** | 1.0 | Primary supervision |
| **L_coord** | 0.5 | Refinement (not essential) |
| **L_anatomical** | 0.1 | Regularization, not strong constraint |
| **L_uncertainty** | 0.1 | Calibration (optional) |

## Training Tips

1. **Start simple**: Use only heatmap loss initially
2. **Add gradually**: Add coordinate loss after heatmap converges
3. **Watch anatomical loss**: If too high, predictions may look unnatural
4. **Monitor components**: Track individual losses to debug training

## References

1. Wu, W., et al. (2018). Look at Boundary: A Boundary-Aware Face Alignment Algorithm. CVPR. (Wing Loss)
2. Wang, X., et al. (2019). Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression. ICCV.
3. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning? NeurIPS.
