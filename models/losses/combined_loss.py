"""
Combined Multi-Component Loss for MAHT-Net

Combines multiple loss functions for comprehensive training:
1. Heatmap Loss: MSE or Focal MSE for heatmap regression
2. Coordinate Loss: Wing loss for direct coordinate supervision
3. Anatomical Loss: Structure constraints (optional)

Reference:
- Wu et al. (2018) Wing Loss for Robust Facial Landmark Localization
- Lin et al. (2017) Focal Loss for Dense Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .anatomical_loss import AnatomicalStructureLoss


class WingLoss(nn.Module):
    """
    Wing Loss for keypoint localization.
    
    Proposed by Wu et al. (CVPR 2018) for facial landmark detection.
    Behaves like logarithm for small errors (more sensitive) and
    linear for large errors (prevents outlier domination).
    
    Wing(x) = w * ln(1 + |x|/ε) if |x| < w
              |x| - C          otherwise
    
    where C = w - w * ln(1 + w/ε)
    
    Args:
        w: Wing width (switches from log to linear at this error)
        epsilon: Curvature parameter
    """
    
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Wing loss.
        
        Args:
            pred: (B, K, 2) predicted coordinates
            target: (B, K, 2) ground truth coordinates
            mask: (B, K) validity mask (optional)
            
        Returns:
            loss: Scalar tensor
        """
        diff = torch.abs(pred - target)
        
        # Wing loss formula
        flag = diff < self.w
        loss = torch.where(
            flag,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, K, 1)
            loss = loss * mask
            return loss.sum() / (mask.sum() * 2 + 1e-6)
        
        return loss.mean()


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for heatmap regression.
    
    Addresses class imbalance where most pixels are background.
    Down-weights easy (background) examples and focuses on hard examples.
    
    Loss = (1 - p)^γ * MSE(p, y)
    
    where p is the predicted probability (heatmap value).
    
    Args:
        gamma: Focusing parameter (0 = standard MSE)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal MSE loss.
        
        Args:
            pred: (B, K, H, W) predicted heatmaps
            target: (B, K, H, W) ground truth heatmaps
            
        Returns:
            loss: Scalar tensor
        """
        # Standard MSE
        mse = (pred - target) ** 2
        
        # Focal weight: (1 - target)^gamma
        # Higher weight for pixels with high target value (keypoint locations)
        weight = (1 - target) ** self.gamma
        
        # Inverse weight for keypoint regions (where target > 0)
        # This makes it focus MORE on keypoint regions
        focal_weight = torch.where(
            target > 0.01,
            target ** (self.gamma / 2),  # Higher weight for keypoints
            weight
        )
        
        loss = focal_weight * mse
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss (AWing) - improved version of Wing Loss.
    
    From Wang et al. (ICCV 2019) "Adaptive Wing Loss for Robust
    Face Alignment via Heatmap Regression"
    
    Better suited for heatmap regression than standard Wing loss.
    """
    
    def __init__(
        self,
        omega: float = 14.0,
        epsilon: float = 1.0,
        alpha: float = 2.1,
        theta: float = 0.5
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.alpha = alpha
        self.theta = theta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute AWing loss."""
        delta = (target - pred).abs()
        
        A = self.omega * (
            1 / (1 + (self.theta / self.epsilon) ** (self.alpha - target))
        ) * (self.alpha - target) * \
            ((self.theta / self.epsilon) ** (self.alpha - target - 1)) * \
            (1 / self.epsilon)
        
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - target)
        )
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(
                1 + (delta / self.epsilon) ** (self.alpha - target)
            ),
            A * delta - C
        )
        
        return loss.mean()


class MAHTNetLoss(nn.Module):
    """
    Combined loss function for MAHT-Net.
    
    L_total = λ₁·L_heatmap + λ₂·L_coordinate + λ₃·L_anatomical
    
    Args:
        heatmap_loss_type: 'mse' or 'focal_mse'
        coord_loss_type: 'wing' or 'l1'
        use_anatomical_loss: Whether to include anatomical constraints
        lambda_heatmap: Weight for heatmap loss
        lambda_coord: Weight for coordinate loss
        lambda_anatomical: Weight for anatomical loss
    """
    
    def __init__(
        self,
        heatmap_loss_type: str = 'mse',
        coord_loss_type: str = 'wing',
        use_anatomical_loss: bool = True,
        lambda_heatmap: float = 1.0,
        lambda_coord: float = 0.5,
        lambda_anatomical: float = 0.1,
        view: str = 'AP'
    ):
        super().__init__()
        
        self.lambda_heatmap = lambda_heatmap
        self.lambda_coord = lambda_coord
        self.lambda_anatomical = lambda_anatomical
        self.use_anatomical_loss = use_anatomical_loss
        self.view = view
        
        # Heatmap loss
        if heatmap_loss_type == 'focal_mse':
            self.heatmap_loss = FocalMSELoss(gamma=2.0)
        else:
            self.heatmap_loss = nn.MSELoss()
        
        # Coordinate loss
        if coord_loss_type == 'wing':
            self.coord_loss = WingLoss(w=10.0, epsilon=2.0)
        else:
            self.coord_loss = nn.L1Loss()
        
        # Anatomical loss
        if use_anatomical_loss:
            self.anatomical_loss = AnatomicalStructureLoss()
        
        print(f"  ✓ MAHTNetLoss: heatmap={heatmap_loss_type}, coord={coord_loss_type}, anatomical={use_anatomical_loss}")
    
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        pred_keypoints: Optional[torch.Tensor] = None,
        gt_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred_heatmaps: (B, K, H, W) predicted heatmaps
            gt_heatmaps: (B, K, H, W) ground truth heatmaps
            pred_keypoints: (B, K, 2) predicted coordinates (optional)
            gt_keypoints: (B, K, 2) ground truth coordinates (optional)
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_heatmaps.device)
        
        # 1. Heatmap loss (always computed)
        loss_heatmap = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
        loss_dict['heatmap'] = loss_heatmap
        total_loss = total_loss + self.lambda_heatmap * loss_heatmap
        
        # 2. Coordinate loss (if coordinates provided)
        if pred_keypoints is not None and gt_keypoints is not None:
            loss_coord = self.coord_loss(pred_keypoints, gt_keypoints)
            loss_dict['coordinate'] = loss_coord
            total_loss = total_loss + self.lambda_coord * loss_coord
        
        # 3. Anatomical loss (if enabled and coordinates provided)
        if self.use_anatomical_loss and pred_keypoints is not None:
            loss_anat, anat_dict = self.anatomical_loss(pred_keypoints, self.view)
            loss_dict['anatomical'] = loss_anat
            loss_dict.update(anat_dict)
            total_loss = total_loss + self.lambda_anatomical * loss_anat
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def extract_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Simple argmax extraction for computing coordinate loss during training.
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        
    Returns:
        keypoints: (B, K, 2) coordinates
    """
    B, K, H, W = heatmaps.shape
    
    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)
    
    # Find argmax positions
    max_idx = heatmaps_flat.argmax(dim=2)  # (B, K)
    
    # Convert to (x, y) coordinates
    y = (max_idx // W).float()
    x = (max_idx % W).float()
    
    return torch.stack([x, y], dim=2)  # (B, K, 2)


def test_combined_loss():
    """Test combined loss functions."""
    print("\n" + "="*60)
    print("Testing Combined Loss Functions")
    print("="*60)
    
    B, K, H, W = 2, 20, 512, 512
    
    # Create dummy predictions and targets
    pred_heatmaps = torch.randn(B, K, H, W).sigmoid()  # Sigmoid for valid heatmaps
    gt_heatmaps = torch.randn(B, K, H, W).sigmoid()
    
    pred_keypoints = torch.rand(B, K, 2) * 512
    gt_keypoints = torch.rand(B, K, 2) * 512
    
    print("\n1. Testing Wing Loss...")
    wing = WingLoss()
    loss_wing = wing(pred_keypoints, gt_keypoints)
    print(f"   Wing Loss: {loss_wing.item():.4f}")
    print("   ✓ Wing Loss computed")
    
    print("\n2. Testing Focal MSE Loss...")
    focal = FocalMSELoss()
    loss_focal = focal(pred_heatmaps, gt_heatmaps)
    print(f"   Focal MSE: {loss_focal.item():.4f}")
    print("   ✓ Focal MSE computed")
    
    print("\n3. Testing Combined MAHTNet Loss...")
    combined = MAHTNetLoss(
        heatmap_loss_type='mse',
        coord_loss_type='wing',
        use_anatomical_loss=True
    )
    
    total, loss_dict = combined(
        pred_heatmaps, gt_heatmaps,
        pred_keypoints, gt_keypoints
    )
    
    print(f"   Total Loss: {total.item():.4f}")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"   {name}: {value.item():.4f}")
    print("   ✓ Combined Loss computed")
    
    print("\n4. Testing without coordinate loss...")
    total_no_coord, _ = combined(pred_heatmaps, gt_heatmaps)
    print(f"   Total (heatmap only): {total_no_coord.item():.4f}")
    print("   ✓ Heatmap-only mode works")
    
    print("\n✓ All loss function tests passed!")


if __name__ == "__main__":
    test_combined_loss()
