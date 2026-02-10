"""
MAHT-Net Components

This package contains the modular components of MAHT-Net:
- efficientnet_backbone: Multi-scale feature extraction with EfficientNetV2-S
- decoder: Multi-scale decoder with skip connections for heatmap generation
- transformer_bridge: Global context modeling (Phase 2)
- vam: Vertebral Attention Module (Phase 2)
- uncertainty: Uncertainty estimation heads (Phase 3)
"""

from .efficientnet_backbone import EfficientNetV2Backbone
from .decoder import MultiScaleDecoder, SimpleDecoder
from .transformer_bridge import TransformerBridge
from .vam import VertebralAttentionModule, VAM
from .uncertainty import (
    UncertaintyHead,
    HeatmapSpreadUncertainty,
    CombinedUncertainty,
    NLLLoss
)

__all__ = [
    'EfficientNetV2Backbone',
    'MultiScaleDecoder',
    'SimpleDecoder',
    'TransformerBridge',
    'VertebralAttentionModule',
    'VAM',
    'UncertaintyHead',
    'HeatmapSpreadUncertainty',
    'CombinedUncertainty',
    'NLLLoss',
]
