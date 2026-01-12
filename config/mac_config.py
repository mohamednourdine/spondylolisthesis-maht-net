"""
Mac-optimized configuration for local training.
Uses smaller images and minimal augmentation for faster training.
"""

from pathlib import Path


class MacConfig:
    """Configuration optimized for Mac local training."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / 'data' / 'Train' / 'Keypointrcnn_data'
    EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
    RESULTS_DIR = EXPERIMENTS_DIR / 'results'
    
    # Data paths
    TRAIN_IMAGE_DIR = DATA_ROOT / 'images' / 'train'
    TRAIN_LABEL_DIR = DATA_ROOT / 'labels' / 'train'
    VAL_IMAGE_DIR = DATA_ROOT / 'images' / 'val'
    VAL_LABEL_DIR = DATA_ROOT / 'labels' / 'val'
    
    # Test data paths
    TEST_ROOT = PROJECT_ROOT / 'data' / 'Test'
    TEST_IMAGE_DIR = TEST_ROOT / 'images'
    TEST_LABEL_DIR = TEST_ROOT / 'labels'
    
    # ===== MAC OPTIMIZATIONS =====
    
    # Full resolution for better accuracy
    IMAGE_SIZE = (512, 512)  # Full resolution
    
    # Enable augmentation for better generalization
    USE_AUGMENTATION = False  # Full augmentation (too slow)
    USE_LIGHT_AUGMENTATION = True  # Light augmentation (flips/rotations/contrast)
    
    # Smaller batch size for memory (512x512 needs less batch size)
    BATCH_SIZE = 2  # Optimal for 512x512 images on Mac (4x faster than batch_size=8)
    NUM_WORKERS = 0  # 0 is often faster on Mac
    
    # More epochs for full training
    NUM_EPOCHS = 50  # Full training
    
    # Device - Auto-detect MPS (Apple Silicon) or CPU
    DEVICE = 'auto'  # Will detect MPS/CPU automatically
    
    # ===== MODEL SETTINGS =====
    MODEL_NAME = 'unet'
    IN_CHANNELS = 3
    MAX_VERTEBRAE = 10  # Maximum number of vertebrae per image
    CORNERS_PER_VERTEBRA = 4  # 4 corner keypoints per vertebra
    NUM_KEYPOINTS = MAX_VERTEBRAE * CORNERS_PER_VERTEBRA  # 40 output channels
    BILINEAR = True  # Faster than transposed conv
    BASE_CHANNELS = 64  # Full size model (17.27M params)
    
    # Heatmap settings - CRITICAL: Amplitude must match model output range!
    # For 512x512 images, sigma=10.0 matches old project's relative sigma (5.0 for 256x256)
    # UNet without final activation produces outputs roughly [-10, 200] range
    # Using amplitude=10.0 (instead of 1000) allows model to match target scale
    HEATMAP_SIGMA = 10.0  # Scaled from old project: 5.0 * (512/256) = 10.0
    HEATMAP_AMPLITUDE = 10.0  # Reduced from 1000 to match UNet output range
    OUTPUT_STRIDE = 1
    
    # ===== TRAINING SETTINGS =====
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Loss function ('focal', 'adaptive_wing', 'combined', 'weighted_mse')
    # weighted_mse: Simple and effective for heatmaps (RECOMMENDED FOR NOW)
    # focal: CornerNet-style, good for object detection
    # adaptive_wing: Complex, needs careful parameter tuning for heatmaps
    # combined: AWing + MSE, more robust
    LOSS_FUNCTION = 'weighted_mse'
    
    # Focal loss parameters (if using focal)
    FOCAL_ALPHA = 2.0
    FOCAL_BETA = 4.0
    
    # Adaptive Wing parameters (if using adaptive_wing)
    AWING_OMEGA = 14
    AWING_THETA = 0.5
    AWING_EPSILON = 1
    AWING_ALPHA = 2.1
    
    # Scheduler (matching old project)
    SCHEDULER = 'ReduceLROnPlateau'
    SCHEDULER_PATIENCE = 15  # Number of epochs with no improvement before reducing LR
    SCHEDULER_FACTOR = 0.5  # Factor by which to reduce LR
    SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10  # Reduced
    SAVE_CHECKPOINT_EVERY = 5
    
    # Preprocessing
    NORMALIZE = True
    APPLY_CLAHE = False  # Disable for speed
    
    # Random seed
    RANDOM_SEED = 42
    
    @classmethod
    def get_device(cls):
        """Auto-detect best available device."""
        import torch
        if cls.DEVICE != 'auto':
            return cls.DEVICE
        
        # Check for MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            return 'mps'
        # Check for CUDA (unlikely on Mac, but check anyway)
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
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
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            'image_size': cls.IMAGE_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'device': cls.get_device(),
            'base_channels': cls.BASE_CHANNELS,
            'use_augmentation': cls.USE_AUGMENTATION,
        }
