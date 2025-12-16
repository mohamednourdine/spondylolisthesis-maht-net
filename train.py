"""
Main training script - Entry point for all model training.

Usage:
    python train.py --model unet --config config/unet_config.py
    python train.py --model maht-net --config experiments/configs/maht_net_config.yaml
    python train.py --model resnet-keypoint --epochs 100 --batch-size 32
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_registry import ModelRegistry
from config.base_config import BaseConfig
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import SpondylolisthesisAugmentation


def load_config_from_python(config_path: Path):
    """Load configuration from Python file."""
    # Simply import from config module
    config_name = config_path.stem  # e.g., 'unet_config'
    
    if config_name == 'unet_config':
        from config.unet_config import UNetConfig
        return UNetConfig
    elif config_name == 'base_config':
        from config.base_config import BaseConfig
        return BaseConfig
    else:
        # Fallback: try dynamic import
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules['config_module'] = config_module
        spec.loader.exec_module(config_module)
        
        # Get the config class
        config_classes = [
            getattr(config_module, name) for name in dir(config_module)
            if isinstance(getattr(config_module, name), type) and 
            name.endswith('Config') and name != 'BaseConfig'
        ]
        
        if not config_classes:
            raise ValueError(f"No config class found in {config_path}")
        
        return config_classes[0]


def setup_training(args):
    """
    Setup training components based on model and config.
    
    Returns:
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, config
    """
    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix == '.yaml':
            config_dict = BaseConfig.from_yaml(config_path)
            Config = type('DynamicConfig', (BaseConfig,), config_dict)
        else:  # Python file
            Config = load_config_from_python(config_path)
    else:
        # Try to load model-specific config
        config_file = PROJECT_ROOT / 'config' / f'{args.model.replace("-", "_")}_config.py'
        if config_file.exists():
            Config = load_config_from_python(config_file)
        else:
            Config = BaseConfig
    
    # Override config with command line arguments
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # Set device - prioritize CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nDevice: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: {device}")
        print("⚠️  CUDA not available - using CPU (training will be slow)")
    
    # Set random seeds
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=Config.IMAGE_SIZE,
        normalize=Config.NORMALIZE,
        apply_clahe=Config.APPLY_CLAHE
    )
    
    # Create augmentation
    augmentation = None
    if hasattr(Config, 'USE_AUGMENTATION') and Config.USE_AUGMENTATION:
        augmentation = SpondylolisthesisAugmentation(mode='train')
    
    # Create dataloaders based on model type
    if args.model == 'unet':
        from src.data.unet_dataset import create_unet_dataloaders
        
        train_loader, val_loader = create_unet_dataloaders(
            train_image_dir=Config.TRAIN_IMAGE_DIR,
            train_label_dir=Config.TRAIN_LABEL_DIR,
            val_image_dir=Config.VAL_IMAGE_DIR,
            val_label_dir=Config.VAL_LABEL_DIR,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            heatmap_sigma=Config.HEATMAP_SIGMA,
            output_stride=Config.OUTPUT_STRIDE,
            preprocessor=preprocessor,
            augmentation=augmentation
        )
    else:
        # Default dataloader for other models
        from src.data.dataset import create_dataloaders
        
        train_loader, val_loader = create_dataloaders(
            train_image_dir=Config.TRAIN_IMAGE_DIR,
            train_label_dir=Config.TRAIN_LABEL_DIR,
            val_image_dir=Config.VAL_IMAGE_DIR,
            val_label_dir=Config.VAL_LABEL_DIR,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            preprocessor=preprocessor,
            augmentation=augmentation
        )
    
    print(f"\nData loaded:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print(f"  Batch size:    {Config.BATCH_SIZE}")
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model_kwargs = {}
    if hasattr(Config, 'get_model_config'):
        model_kwargs = Config.get_model_config()
    
    model = ModelRegistry.create(args.model, **model_kwargs)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    if args.model == 'unet':
        from training.losses import UNetKeypointLoss
        loss_kwargs = Config.get_loss_config() if hasattr(Config, 'get_loss_config') else {}
        criterion = UNetKeypointLoss(**loss_kwargs)
    else:
        from training.losses import MSELoss
        criterion = MSELoss()
    
    # Create optimizer
    optimizer_name = Config.OPTIMIZER if hasattr(Config, 'OPTIMIZER') else 'Adam'
    optimizer_kwargs = Config.get_optimizer_config() if hasattr(Config, 'get_optimizer_config') else {
        'lr': Config.LEARNING_RATE,
        'weight_decay': Config.WEIGHT_DECAY
    }
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"\nOptimizer: {optimizer_name}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    
    # Create scheduler
    scheduler = None
    scheduler_name = Config.SCHEDULER if hasattr(Config, 'SCHEDULER') else None
    if scheduler_name:
        scheduler_kwargs = Config.get_scheduler_config() if hasattr(Config, 'get_scheduler_config') else {}
        
        if scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        
        print(f"Scheduler: {scheduler_name}")
    
    # Convert config to dict for saving
    config_dict = Config.to_dict() if hasattr(Config, 'to_dict') else {}
    
    return model, train_loader, val_loader, criterion, optimizer, scheduler, device, config_dict, Config


def main():
    parser = argparse.ArgumentParser(description='Train models for spondylolisthesis detection')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=ModelRegistry.list_models(),
                       help='Model to train')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (.py or .yaml)')
    
    # Training hyperparameters (override config)
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # Experiment naming
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Spondylolisthesis Detection - Model Training")
    print("="*60)
    print(f"Model: {args.model}")
    
    # Setup training
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, config_dict, Config = setup_training(args)
    
    # Add model name to config for better organization
    config_dict['model_name'] = args.model
    
    # Create save directory
    experiment_name = args.experiment_name or f"{args.model}_training"
    save_dir = Config.RESULTS_DIR
    
    # Create trainer based on model type
    if args.model == 'unet':
        from training.unet_trainer import UNetTrainer
        trainer = UNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config_dict,
            save_dir=save_dir,
            scheduler=scheduler,
            experiment_name=experiment_name
        )
    else:
        # For other models, use base trainer (to be extended)
        from training.base_trainer import BaseTrainer
        raise NotImplementedError(f"Trainer for {args.model} not yet implemented")
    
    # Train
    resume_from = Path(args.resume) if args.resume else None
    trainer.train(Config.NUM_EPOCHS, resume_from=resume_from)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
