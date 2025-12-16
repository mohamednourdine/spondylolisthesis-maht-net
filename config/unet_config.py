"""
UNet-specific configuration.
"""

from .base_config import BaseConfig


class UNetConfig(BaseConfig):
    """Configuration for UNet model."""
    
    # Model architecture
    MODEL_NAME = 'unet'
    IN_CHANNELS = 3
    NUM_KEYPOINTS = 4  # 4 corners per vertebra
    BILINEAR = False
    BASE_CHANNELS = 64
    
    # Heatmap generation
    HEATMAP_SIGMA = 3.0
    OUTPUT_STRIDE = 1
    
    # Loss function
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 2.0
    FOCAL_BETA = 4.0
    
    # Optimizer
    OPTIMIZER = 'Adam'  # 'Adam' or 'SGD'
    MOMENTUM = 0.9  # For SGD
    
    # Learning rate scheduler
    SCHEDULER = 'StepLR'  # 'StepLR', 'ReduceLROnPlateau', or None
    STEP_SIZE = 10  # For StepLR
    GAMMA = 0.1  # For StepLR
    LR_PATIENCE = 5  # For ReduceLROnPlateau
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_MODE = 'train'
    
    # Training specifics
    BATCH_SIZE = 8  # Reduced for UNet (memory intensive)
    NUM_EPOCHS = 50
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model-specific configuration."""
        return {
            'in_channels': cls.IN_CHANNELS,
            'num_keypoints': cls.NUM_KEYPOINTS,
            'bilinear': cls.BILINEAR,
            'base_channels': cls.BASE_CHANNELS
        }
    
    @classmethod
    def get_loss_config(cls) -> dict:
        """Get loss function configuration."""
        return {
            'use_focal': cls.USE_FOCAL_LOSS,
            'focal_alpha': cls.FOCAL_ALPHA,
            'focal_beta': cls.FOCAL_BETA
        }
    
    @classmethod
    def get_optimizer_config(cls) -> dict:
        """Get optimizer configuration."""
        config = {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY
        }
        if cls.OPTIMIZER == 'SGD':
            config['momentum'] = cls.MOMENTUM
        return config
    
    @classmethod
    def get_scheduler_config(cls) -> dict:
        """Get scheduler configuration."""
        if cls.SCHEDULER == 'StepLR':
            return {
                'step_size': cls.STEP_SIZE,
                'gamma': cls.GAMMA
            }
        elif cls.SCHEDULER == 'ReduceLROnPlateau':
            return {
                'patience': cls.LR_PATIENCE,
                'factor': cls.GAMMA,
                'mode': 'min'
            }
        return {}
