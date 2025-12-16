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
    Combines MSE loss with optional focal loss for better convergence.
    """
    
    def __init__(self, use_focal=True, focal_alpha=2.0, focal_beta=4.0):
        super(UNetKeypointLoss, self).__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
        """
        if self.use_focal:
            # Focal loss for keypoint detection
            # Helps focus on hard examples
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            
            # Positive focal loss
            pos_loss = -self.focal_alpha * torch.pow(1 - pred_sigmoid, self.focal_beta) * \
                       target_heatmaps * torch.log(pred_sigmoid + 1e-8)
            
            # Negative focal loss
            neg_loss = -(1 - self.focal_alpha) * torch.pow(pred_sigmoid, self.focal_beta) * \
                       (1 - target_heatmaps) * torch.log(1 - pred_sigmoid + 1e-8)
            
            loss = (pos_loss + neg_loss).mean()
        else:
            # Simple MSE loss with sigmoid
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            loss = self.mse(pred_sigmoid, target_heatmaps)
        
        return loss