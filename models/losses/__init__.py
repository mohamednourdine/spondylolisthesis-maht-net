"""
MAHT-Net Loss Functions

This package contains all loss functions used in MAHT-Net training:
- Heatmap losses (MSE, Focal MSE)
- Wing loss for coordinate supervision
- Anatomical structure loss (our contribution)
- Combined multi-component loss
"""

from .anatomical_loss import (
    AnatomicalStructureLoss,
    VerticalOrderingLoss,
    EdgeParallelismLoss,
    AspectRatioLoss,
)
from .combined_loss import MAHTNetLoss, WingLoss, FocalMSELoss

__all__ = [
    'AnatomicalStructureLoss',
    'VerticalOrderingLoss',
    'EdgeParallelismLoss',
    'AspectRatioLoss',
    'MAHTNetLoss',
    'WingLoss',
    'FocalMSELoss',
]
