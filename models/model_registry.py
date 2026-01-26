"""
Model registry for easy model creation and management.
"""

from typing import Dict, Any, Callable
import torch.nn as nn


class ModelRegistry:
    """Registry for all available models."""
    
    _models: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model factory function.
        
        Usage:
            @ModelRegistry.register('unet')
            def create_unet(**kwargs):
                return UNet(**kwargs)
        """
        def decorator(func: Callable):
            cls._models[name.lower()] = func
            return func
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Create a model by name.
        
        Args:
            name: Model name
            **kwargs: Model-specific arguments
            
        Returns:
            Instantiated model
        """
        name = name.lower()
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls):
        """List all registered models."""
        return list(cls._models.keys())


# Register models
from .unet import create_unet
from .resnet_heatmap import create_resnet_heatmap
from .hrnet_heatmap import create_hrnet_heatmap

@ModelRegistry.register('unet')
def create_unet_model(**kwargs):
    """Create UNet model."""
    return create_unet(**kwargs)


@ModelRegistry.register('resnet')
def create_resnet_model(**kwargs):
    """Create ResNet-50 heatmap model."""
    return create_resnet_heatmap(**kwargs)


@ModelRegistry.register('resnet-heatmap')
def create_resnet_heatmap_model(**kwargs):
    """Create ResNet-50 heatmap model (alias)."""
    return create_resnet_heatmap(**kwargs)


@ModelRegistry.register('hrnet')
def create_hrnet_model(**kwargs):
    """Create HRNet-W32 heatmap model."""
    return create_hrnet_heatmap(**kwargs)


@ModelRegistry.register('hrnet-w32')
def create_hrnet_w32_model(**kwargs):
    """Create HRNet-W32 heatmap model (alias)."""
    return create_hrnet_heatmap(**kwargs)


@ModelRegistry.register('hrnet-multiscale')
def create_hrnet_multiscale_model(**kwargs):
    """Create HRNet-W32 with multi-scale fusion."""
    return create_hrnet_heatmap(multi_scale=True, **kwargs)


# Placeholder registrations for future models
@ModelRegistry.register('maht-net')
def create_maht_net_model(**kwargs):
    """Create MAHT-Net model (to be implemented)."""
    raise NotImplementedError("MAHT-Net not yet implemented")


@ModelRegistry.register('resnet-keypoint')
def create_resnet_keypoint_model(**kwargs):
    """Create ResNet keypoint model (to be implemented)."""
    raise NotImplementedError("ResNet-Keypoint not yet implemented")


@ModelRegistry.register('keypoint-rcnn')
def create_keypoint_rcnn_model(**kwargs):
    """Create Keypoint R-CNN model (to be implemented)."""
    raise NotImplementedError("Keypoint R-CNN not yet implemented")
