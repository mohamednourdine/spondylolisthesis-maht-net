"""
Loss functions for different models.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error Loss."""
    
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.mse(predictions, targets)


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss."""
    
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        return self.ce(predictions, targets)


class CustomLoss(nn.Module):
    """Combined MSE and Cross Entropy Loss."""
    
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = MSELoss()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.alpha * mse + (1 - self.alpha) * ce


class UNetKeypointLoss(nn.Module):
    """
    Loss function for UNet keypoint detection.
    Uses focal loss variant for heatmap regression (CornerNet-style).
    
    Reference: CornerNet: Detecting Objects as Paired Keypoints
    https://arxiv.org/abs/1808.01244
    """
    
    def __init__(self, use_focal=True, focal_alpha=2.0, focal_beta=4.0):
        """
        Args:
            use_focal: Whether to use focal loss (recommended)
            focal_alpha: Focusing parameter for hard examples (default: 2.0)
            focal_beta: Penalty reduction for easy negatives near keypoints (default: 4.0)
        """
        super(UNetKeypointLoss, self).__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha  # Focusing parameter (gamma in original paper)
        self.focal_beta = focal_beta    # Penalty reduction for near-keypoint negatives
        self.mse = nn.MSELoss()
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
        """
        if self.use_focal:
            # CornerNet-style focal loss for heatmap regression
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            
            # Clamp predictions for numerical stability
            pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-4, max=1-1e-4)
            
            # Find positive locations (keypoints)
            pos_mask = target_heatmaps.eq(1).float()
            neg_mask = target_heatmaps.lt(1).float()
            
            # Positive loss: focus on hard-to-detect positives
            pos_loss = torch.pow(1 - pred_sigmoid, self.focal_alpha) * \
                       torch.log(pred_sigmoid) * pos_mask
            
            # Negative loss: reduce penalty near keypoints using target values
            neg_loss = torch.pow(1 - target_heatmaps, self.focal_beta) * \
                       torch.pow(pred_sigmoid, self.focal_alpha) * \
                       torch.log(1 - pred_sigmoid) * neg_mask
            
            # Combine losses
            num_pos = pos_mask.sum()
            if num_pos == 0:
                loss = -neg_loss.sum()
            else:
                loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
            
            return loss
        else:
            # Simple MSE loss with sigmoid
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            loss = self.mse(pred_sigmoid, target_heatmaps)
            return loss


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap-based landmark detection.
    
    Better than Focal Loss for precise landmark localization because:
    - Focuses more on small/medium errors (more important for landmarks)
    - Reduces influence of large background errors
    - Adaptive behavior near ground truth vs far from it
    
    Reference: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    https://arxiv.org/abs/1904.07399
    
    This loss has shown superior performance for medical landmark detection
    compared to MSE and Focal Loss.
    """
    
    def __init__(self, omega=10, theta=0.5, epsilon=0.5, alpha=2.1):
        """
        Args:
            omega: Controls the range of non-linear part (default: 10 for heatmaps)
            theta: Threshold between linear and non-linear parts (default: 0.5)
            epsilon: Small constant for numerical stability (default: 0.5 for heatmaps)
            alpha: Controls the shape of the non-linear region (default: 2.1)
        
        Note: Parameters are tuned for heatmap regression (values in [0,1])
              Original paper uses different values for pixel coordinate regression.
        """
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
        
        MPS-compatible version using smooth absolute value (no sgn in backward).
        """
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred_heatmaps)
        
        # Calculate SMOOTH absolute difference (MPS-compatible)
        # Instead of abs(x), use sqrt(x^2 + eps) which has smooth gradients
        raw_diff = target_heatmaps - pred_sigmoid
        smooth_abs_eps = 1e-6
        diff = torch.sqrt(raw_diff * raw_diff + smooth_abs_eps)
        
        # Pre-compute constants as scalars
        theta_over_eps = self.theta / self.epsilon
        theta_over_eps_alpha_minus_2 = theta_over_eps ** (self.alpha - 2)
        theta_over_eps_alpha_minus_1 = theta_over_eps ** (self.alpha - 1)
        
        A = self.omega * (1 / (1 + theta_over_eps_alpha_minus_2)) * \
            (self.alpha - 2) * theta_over_eps_alpha_minus_1 / self.epsilon
        C = self.omega * (1 - theta_over_eps_alpha_minus_2)
        
        # Calculate both loss terms
        # Small errors: omega * ln(1 + (|diff|/epsilon)^alpha)
        diff_normalized = diff / self.epsilon
        small_error_loss = self.omega * torch.log1p(torch.pow(diff_normalized, self.alpha))
        
        # Large errors: A*|diff| - C
        large_error_loss = A * diff - C
        
        # Smooth transition using sigmoid (MPS-compatible)
        temperature = self.theta * 0.1
        transition_weight = torch.sigmoid((self.theta - diff) / temperature)
        
        # Smooth blend between the two losses
        loss = transition_weight * small_error_loss + (1 - transition_weight) * large_error_loss
        
        # Weight by target values (focus on keypoint regions)
        weights = 1.0 + target_heatmaps
        loss = loss * weights
        
        return loss.mean()


class MSEWithWeightedBackground(nn.Module):
    """
    MSE Loss with reduced weight for background pixels.
    Simple but effective for heatmap regression.
    
    NOTE: Does NOT use sigmoid on predictions (matches working old project).
    Model outputs raw heatmaps directly.
    """
    
    def __init__(self, background_weight=0.1):
        """
        Args:
            background_weight: Weight for background pixels (default: 0.1)
        """
        super(MSEWithWeightedBackground, self).__init__()
        self.background_weight = background_weight
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W] (raw values, no sigmoid)
            target_heatmaps: Target heatmaps [B, K, H, W]
        """
        # NO SIGMOID - use raw predictions (like old working project)
        # Calculate squared error
        squared_error = (pred_heatmaps - target_heatmaps) ** 2
        
        # Create weight map: low weight for background, high weight for keypoints
        weights = torch.where(
            target_heatmaps > 0.1,  # Near keypoints
            torch.ones_like(target_heatmaps),
            torch.full_like(target_heatmaps, self.background_weight)
        )
        
        # Weighted MSE
        weighted_error = squared_error * weights
        
        return weighted_error.mean()


class CombinedKeypointLoss(nn.Module):
    """
    Combined loss: Adaptive Wing + MSE
    - Adaptive Wing for precise localization
    - MSE for overall heatmap quality
    
    This combination often works best in practice.
    """
    
    def __init__(self, awing_weight=0.7, mse_weight=0.3):
        """
        Args:
            awing_weight: Weight for Adaptive Wing loss (default: 0.7)
            mse_weight: Weight for MSE loss (default: 0.3)
        """
        super(CombinedKeypointLoss, self).__init__()
        self.awing = AdaptiveWingLoss()
        self.mse = MSEWithWeightedBackground()
        self.awing_weight = awing_weight
        self.mse_weight = mse_weight
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
        """
        awing_loss = self.awing(pred_heatmaps, target_heatmaps)
        mse_loss = self.mse(pred_heatmaps, target_heatmaps)
        
        total_loss = self.awing_weight * awing_loss + self.mse_weight * mse_loss
        
        return total_loss