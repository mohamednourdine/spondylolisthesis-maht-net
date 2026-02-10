"""
Anatomical Structure Loss

Novel loss function that penalizes anatomically impossible predictions.
This encodes domain knowledge about vertebral structure directly into training.

Constraints enforced:
1. Vertical ordering: L1 should be above L2, L2 above L3, etc.
2. Edge parallelism: Top and bottom edges of same vertebra roughly parallel
3. Aspect ratio: Vertebrae have expected width/height ratios

Reference:
- Wu et al. (2018) - Anatomy-aware 3D human pose estimation
- Our contribution: Vertebra-specific constraints for BUU-LSPINE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class VerticalOrderingLoss(nn.Module):
    """
    Enforce vertical ordering constraint.
    
    Penalizes cases where:
    - L2-top is above L1-bottom (inverted vertebrae)
    - L3-top is above L2-bottom
    - etc.
    
    This is critical for anatomically valid predictions.
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: Minimum expected gap between vertebrae (in normalized coords)
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, keypoints: torch.Tensor, view: str = 'AP') -> torch.Tensor:
        """
        Compute vertical ordering loss.
        
        Args:
            keypoints: (B, K, 2) predicted keypoint coordinates
            view: 'AP' or 'LA'
            
        Returns:
            loss: Scalar tensor
        """
        B, K, _ = keypoints.shape
        
        # Get y-coordinates (row indices)
        y_coords = keypoints[:, :, 1]  # (B, K)
        
        # Number of edges (rows) 
        # AP: 20 keypoints = 10 rows (5 vertebrae × 2 edges)
        # LA: 22 keypoints = 11 rows (5 vertebrae × 2 edges + 1 S1 edge)
        num_rows = K // 2
        
        # Compute mean y per row (average of left and right corners)
        y_per_row = y_coords.view(B, num_rows, 2).mean(dim=2)  # (B, num_rows)
        
        # Ordering loss: penalize if row i+1 is above row i
        # We expect y to increase (go down) as row index increases
        loss = torch.tensor(0.0, device=keypoints.device)
        
        for i in range(num_rows - 1):
            # Penalize if next row's y is less than current row's y
            diff = y_per_row[:, i] - y_per_row[:, i + 1] + self.margin
            loss = loss + F.relu(diff).mean()
        
        return loss / max(num_rows - 1, 1)


class EdgeParallelismLoss(nn.Module):
    """
    Enforce edge parallelism constraint.
    
    Top and bottom edges of the same vertebra should be roughly parallel.
    This constraint helps maintain realistic vertebra shapes.
    """
    
    def __init__(self, max_angle_deg: float = 15.0):
        """
        Args:
            max_angle_deg: Maximum allowed angle difference in degrees
        """
        super().__init__()
        self.max_angle_rad = max_angle_deg * (3.14159 / 180.0)
    
    def _compute_edge_angle(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute angle of edge from left to right corner."""
        dx = right[:, 0] - left[:, 0]
        dy = right[:, 1] - left[:, 1]
        return torch.atan2(dy, dx)
    
    def forward(self, keypoints: torch.Tensor, view: str = 'AP') -> torch.Tensor:
        """
        Compute edge parallelism loss.
        
        Args:
            keypoints: (B, K, 2) predicted keypoint coordinates
            view: 'AP' or 'LA'
            
        Returns:
            loss: Scalar tensor
        """
        B, K, _ = keypoints.shape
        num_vertebrae = 5 if view.upper() == 'AP' else 5
        
        loss = torch.tensor(0.0, device=keypoints.device)
        
        for v in range(num_vertebrae):
            # Top edge: row v*2 (points v*4 and v*4+1)
            # Bottom edge: row v*2+1 (points v*4+2 and v*4+3)
            top_left_idx = v * 4
            top_right_idx = v * 4 + 1
            bottom_left_idx = v * 4 + 2
            bottom_right_idx = v * 4 + 3
            
            if bottom_right_idx >= K:
                break
            
            # Compute edge angles
            top_angle = self._compute_edge_angle(
                keypoints[:, top_left_idx],
                keypoints[:, top_right_idx]
            )
            bottom_angle = self._compute_edge_angle(
                keypoints[:, bottom_left_idx],
                keypoints[:, bottom_right_idx]
            )
            
            # Penalize angle difference exceeding max
            angle_diff = torch.abs(top_angle - bottom_angle)
            excess = F.relu(angle_diff - self.max_angle_rad)
            loss = loss + excess.mean()
        
        return loss / max(num_vertebrae, 1)


class AspectRatioLoss(nn.Module):
    """
    Enforce vertebra aspect ratio constraint.
    
    Vertebrae have expected width/height ratios that vary by level.
    This constraint prevents unrealistic deformations.
    """
    
    def __init__(
        self,
        expected_ratios: Optional[Dict[str, float]] = None,
        tolerance: float = 0.3
    ):
        """
        Args:
            expected_ratios: Expected width/height ratios per vertebra
            tolerance: Allowed deviation from expected ratio
        """
        super().__init__()
        
        # Default expected ratios (lumbar vertebrae are roughly 1.5-2x wider than tall)
        self.expected_ratios = expected_ratios or {
            'L1': 1.6,
            'L2': 1.6,
            'L3': 1.7,
            'L4': 1.8,
            'L5': 1.9,  # L5 is typically wider
        }
        self.tolerance = tolerance
    
    def forward(self, keypoints: torch.Tensor, view: str = 'AP') -> torch.Tensor:
        """
        Compute aspect ratio loss.
        
        Args:
            keypoints: (B, K, 2) predicted keypoint coordinates
            view: 'AP' or 'LA'
            
        Returns:
            loss: Scalar tensor
        """
        B, K, _ = keypoints.shape
        
        loss = torch.tensor(0.0, device=keypoints.device)
        vertebra_names = ['L1', 'L2', 'L3', 'L4', 'L5']
        
        for v, name in enumerate(vertebra_names):
            # Get 4 corners of vertebra v
            top_left_idx = v * 4
            top_right_idx = v * 4 + 1
            bottom_left_idx = v * 4 + 2
            bottom_right_idx = v * 4 + 3
            
            if bottom_right_idx >= K:
                break
            
            # Compute width (average of top and bottom edge widths)
            top_width = torch.abs(
                keypoints[:, top_right_idx, 0] - keypoints[:, top_left_idx, 0]
            )
            bottom_width = torch.abs(
                keypoints[:, bottom_right_idx, 0] - keypoints[:, bottom_left_idx, 0]
            )
            width = (top_width + bottom_width) / 2
            
            # Compute height (average of left and right side heights)
            left_height = torch.abs(
                keypoints[:, bottom_left_idx, 1] - keypoints[:, top_left_idx, 1]
            )
            right_height = torch.abs(
                keypoints[:, bottom_right_idx, 1] - keypoints[:, top_right_idx, 1]
            )
            height = (left_height + right_height) / 2
            
            # Compute ratio (with small epsilon for numerical stability)
            ratio = width / (height + 1e-6)
            
            # Penalize deviation from expected ratio
            expected = self.expected_ratios.get(name, 1.7)
            deviation = torch.abs(ratio - expected) - self.tolerance
            loss = loss + F.relu(deviation).mean()
        
        return loss / len(vertebra_names)


class AnatomicalStructureLoss(nn.Module):
    """
    Combined anatomical structure loss.
    
    Combines multiple anatomical constraints:
    1. Vertical ordering (most important)
    2. Edge parallelism
    3. Aspect ratio
    
    Args:
        ordering_weight: Weight for vertical ordering loss
        parallelism_weight: Weight for edge parallelism loss
        ratio_weight: Weight for aspect ratio loss
    """
    
    def __init__(
        self,
        ordering_weight: float = 1.0,
        parallelism_weight: float = 0.1,
        ratio_weight: float = 0.1
    ):
        super().__init__()
        
        self.ordering_loss = VerticalOrderingLoss()
        self.parallelism_loss = EdgeParallelismLoss()
        self.ratio_loss = AspectRatioLoss()
        
        self.weights = {
            'ordering': ordering_weight,
            'parallelism': parallelism_weight,
            'ratio': ratio_weight
        }
    
    def forward(
        self,
        keypoints: torch.Tensor,
        view: str = 'AP'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined anatomical loss.
        
        Args:
            keypoints: (B, K, 2) predicted keypoint coordinates
            view: 'AP' or 'LA'
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values for logging
        """
        loss_ordering = self.ordering_loss(keypoints, view)
        loss_parallel = self.parallelism_loss(keypoints, view)
        loss_ratio = self.ratio_loss(keypoints, view)
        
        total_loss = (
            self.weights['ordering'] * loss_ordering +
            self.weights['parallelism'] * loss_parallel +
            self.weights['ratio'] * loss_ratio
        )
        
        loss_dict = {
            'anatomical/ordering': loss_ordering,
            'anatomical/parallelism': loss_parallel,
            'anatomical/ratio': loss_ratio,
            'anatomical/total': total_loss
        }
        
        return total_loss, loss_dict


def test_anatomical_loss():
    """Test anatomical loss functions."""
    print("\n" + "="*60)
    print("Testing Anatomical Structure Loss")
    print("="*60)
    
    # Create properly ordered keypoints (simulate valid predictions)
    B, K = 2, 20
    keypoints = torch.zeros(B, K, 2)
    
    for i in range(K // 2):
        # Each row has left and right corner
        row_y = i * 20 + 100  # Increasing y (going down)
        keypoints[:, i * 2, :] = torch.tensor([[50, row_y]])  # left
        keypoints[:, i * 2 + 1, :] = torch.tensor([[150, row_y]])  # right
    
    # Test each component
    print("\n1. Testing Vertical Ordering Loss...")
    ordering_loss = VerticalOrderingLoss()
    loss_ord = ordering_loss(keypoints)
    print(f"   Loss (valid ordering): {loss_ord.item():.4f}")
    
    # Create invalid ordering
    bad_keypoints = keypoints.clone()
    bad_keypoints[:, 4, 1] = 50  # Move L2 above L1
    loss_ord_bad = ordering_loss(bad_keypoints)
    print(f"   Loss (invalid ordering): {loss_ord_bad.item():.4f}")
    assert loss_ord_bad > loss_ord, "Bad ordering should have higher loss"
    print("   ✓ Ordering loss correctly penalizes inversions")
    
    print("\n2. Testing Edge Parallelism Loss...")
    parallel_loss = EdgeParallelismLoss()
    loss_par = parallel_loss(keypoints)
    print(f"   Loss: {loss_par.item():.4f}")
    print("   ✓ Parallelism loss computed")
    
    print("\n3. Testing Aspect Ratio Loss...")
    ratio_loss = AspectRatioLoss()
    loss_rat = ratio_loss(keypoints)
    print(f"   Loss: {loss_rat.item():.4f}")
    print("   ✓ Aspect ratio loss computed")
    
    print("\n4. Testing Combined Loss...")
    combined_loss = AnatomicalStructureLoss()
    total, loss_dict = combined_loss(keypoints, view='AP')
    print(f"   Total: {total.item():.4f}")
    for name, value in loss_dict.items():
        print(f"   {name}: {value.item():.4f}")
    print("   ✓ Combined loss computed")
    
    print("\n✓ All anatomical loss tests passed!")


if __name__ == "__main__":
    test_anatomical_loss()
