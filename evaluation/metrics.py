"""
Global evaluation metrics for all models.
Provides consistent evaluation across different architectures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_mse(predictions, targets):
    """
    Calculate Mean Squared Error (MSE) - Mean of squared Euclidean distances.
    
    This is reported in many papers alongside MRE. MSE emphasizes larger errors
    more than MRE due to the squaring.
    
    Args:
        predictions: Predicted keypoints [N, 2] or [N, M, 2]
        targets: Target keypoints [N, 2] or [N, M, 2]
        
    Returns:
        Mean squared error in pixels²
    """
    if isinstance(predictions, torch.Tensor):
        squared_distances = ((predictions - targets) ** 2).sum(dim=-1)
        return squared_distances.mean().item()
    else:
        squared_distances = ((predictions - targets) ** 2).sum(axis=-1)
        return float(squared_distances.mean())


def calculate_mre(predictions, targets):
    """
    Calculate Mean Radial Error (MRE) - Euclidean distance between predicted and target keypoints.
    
    Args:
        predictions: Predicted keypoints [N, 2] or [N, M, 2]
        targets: Target keypoints [N, 2] or [N, M, 2]
        
    Returns:
        Mean radial error in pixels
    """
    if isinstance(predictions, torch.Tensor):
        distances = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))
        return distances.mean().item()
    else:
        distances = np.sqrt(((predictions - targets) ** 2).sum(axis=-1))
        return float(distances.mean())


def calculate_sdr(predictions, targets, threshold):
    """
    Calculate Success Detection Rate (SDR) - Percentage of keypoints within threshold.
    
    Args:
        predictions: Predicted keypoints [N, 2] or [N, M, 2]
        targets: Target keypoints [N, 2] or [N, M, 2]
        threshold: Distance threshold in pixels
        
    Returns:
        SDR as a fraction (0-1)
    """
    if isinstance(predictions, torch.Tensor):
        distances = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))
        return (distances <= threshold).float().mean().item()
    else:
        distances = np.sqrt(((predictions - targets) ** 2).sum(axis=-1))
        return float((distances <= threshold).mean())


def calculate_cohens_kappa(predictions, targets):
    """Calculate Cohen's Kappa for classification tasks."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(predictions, targets)


def calculate_accuracy(predictions, targets):
    """Calculate accuracy for classification tasks."""
    if isinstance(predictions, torch.Tensor):
        return (predictions == targets).float().mean().item()
    else:
        return float((predictions == targets).mean())