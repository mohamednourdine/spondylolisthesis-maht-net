"""
Base configuration class with default values.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


class BaseConfig:
    """Base configuration with common settings."""
    
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
    
    # Common hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Image preprocessing
    IMAGE_SIZE = (512, 512)
    NORMALIZE = True
    APPLY_CLAHE = True
    
    # Training
    EARLY_STOPPING_PATIENCE = 15
    SAVE_CHECKPOINT_EVERY = 10
    
    # Device
    DEVICE = 'cuda'  # or 'cpu'
    
    # Random seed
    RANDOM_SEED = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Configuration dictionary
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                # Skip class methods and Path objects
                if isinstance(value, type(cls.to_dict)):
                    continue
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    try:
                        # Test if value is JSON serializable
                        import json
                        json.dumps(value)
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        config_dict[key] = str(value)
        return config_dict
    
    @classmethod
    def update_from_dict(cls, updates: Dict[str, Any]):
        """Update config from dictionary."""
        for key, value in updates.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
