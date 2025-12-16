# Model Training System - Best Practices

## Architecture Overview

This project follows **best practices for multi-model ML projects** with:

### 1. **Modular Architecture**
```
spondylolisthesis-maht-net/
├── config/                    # Configuration management
│   ├── base_config.py        # Base configuration with defaults
│   ├── unet_config.py        # UNet-specific config
│   └── [model]_config.py     # Other model configs
├── models/                    # Model definitions
│   ├── model_registry.py     # Centralized model registry
│   ├── unet.py
│   └── [other models].py
├── training/                  # Training logic
│   ├── base_trainer.py       # Abstract base trainer
│   ├── unet_trainer.py       # Model-specific trainers
│   └── losses.py             # Loss functions
├── train.py                  # Main entry point
└── experiments/              # All experiments and results
    └── results/
        ├── unet_20231215_143022/    # Timestamped experiment
        │   ├── config.json
        │   ├── best_model.pth
        │   ├── training_history.json
        │   └── checkpoint_epoch_*.pth
        └── ...
```

### 2. **Configuration Management**

#### Option A: Python Configuration (Recommended)
```python
# config/unet_config.py
from .base_config import BaseConfig

class UNetConfig(BaseConfig):
    MODEL_NAME = 'unet'
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    # ... model-specific settings
```

#### Option B: YAML Configuration
```yaml
# experiments/configs/unet_config.yaml
model:
  name: "unet"
  in_channels: 3
training:
  batch_size: 8
  num_epochs: 50
```

### 3. **Model Registry Pattern**

Register models once, use anywhere:

```python
from models.model_registry import ModelRegistry

@ModelRegistry.register('unet')
def create_unet_model(**kwargs):
    return UNet(**kwargs)

# Use anywhere in code:
model = ModelRegistry.create('unet', in_channels=3, num_keypoints=4)
```

### 4. **Abstract Base Trainer**

All trainers inherit from `BaseTrainer`:
- ✅ Common training/validation loops
- ✅ Automatic checkpointing
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Metrics tracking
- ✅ Experiment logging

### 5. **Unified Training Entry Point**

## Usage

### Basic Training

```bash
# Train UNet model with default config
python train.py --model unet

# Train with custom config
python train.py --model unet --config config/unet_config.py

# Override hyperparameters
python train.py --model unet --batch-size 16 --epochs 100 --lr 0.0001

# Name your experiment
python train.py --model unet --experiment-name unet_baseline_v1

# Resume from checkpoint
python train.py --model unet --resume experiments/results/unet_*/best_model.pth
```

### Training Other Models (Future)

```bash
python train.py --model maht-net --config config/maht_net_config.py
python train.py --model resnet-keypoint --epochs 100
python train.py --model keypoint-rcnn --batch-size 4
```

### List Available Models

```bash
python -c "from models.model_registry import ModelRegistry; print(ModelRegistry.list_models())"
```

## Adding New Models

### Step 1: Create Model
```python
# models/new_model.py
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # ... implementation

def create_new_model(**kwargs):
    return NewModel(**kwargs)
```

### Step 2: Register Model
```python
# models/model_registry.py
from .new_model import create_new_model

@ModelRegistry.register('new-model')
def create_new_model_registered(**kwargs):
    return create_new_model(**kwargs)
```

### Step 3: Create Config
```python
# config/new_model_config.py
from .base_config import BaseConfig

class NewModelConfig(BaseConfig):
    MODEL_NAME = 'new-model'
    # ... specific settings
```

### Step 4: Create Trainer (if needed)
```python
# training/new_model_trainer.py
from .base_trainer import BaseTrainer

class NewModelTrainer(BaseTrainer):
    def train_epoch(self, epoch):
        # ... implementation
    
    def validate(self, epoch):
        # ... implementation
```

### Step 5: Update train.py
```python
# train.py (in setup_training function)
if args.model == 'new-model':
    from training.new_model_trainer import NewModelTrainer
    # ... setup
```

## Experiment Organization

Each training run creates a timestamped directory:
```
experiments/results/
└── unet_20231215_143022/
    ├── config.json              # Full configuration
    ├── best_model.pth           # Best model checkpoint
    ├── checkpoint_epoch_10.pth  # Periodic checkpoints
    ├── checkpoint_epoch_20.pth
    └── training_history.json    # Loss/metrics history
```

## Features

### ✅ Automatic Features
- Model checkpointing (best + periodic)
- Training history tracking
- Early stopping
- Learning rate scheduling
- GPU/CPU automatic detection
- Random seed setting for reproducibility

### ✅ Flexible Configuration
- Python or YAML configs
- Command-line overrides
- Model-specific settings
- Easy experimentation

### ✅ Extensible
- Add new models easily
- Custom loss functions
- Custom metrics
- Custom trainers

### ✅ Production-Ready
- Proper error handling
- Logging and monitoring
- Resume training capability
- Configuration versioning

## Best Practices Implemented

1. **Separation of Concerns**: Models, training, config are separate
2. **DRY Principle**: Common code in base classes
3. **Registry Pattern**: Easy model management
4. **Factory Pattern**: Flexible object creation
5. **Configuration Management**: Centralized settings
6. **Experiment Tracking**: Automatic versioning and logging
7. **Reproducibility**: Seed setting and config saving
8. **Maintainability**: Clear structure and documentation

## Example: Complete Training Flow

```python
# 1. Define configuration
# config/unet_config.py already exists

# 2. Train model
python train.py --model unet --experiment-name baseline_v1

# 3. Results automatically saved to:
# experiments/results/baseline_v1_20231215_143022/

# 4. Resume if needed:
python train.py --model unet --resume experiments/results/baseline_v1_*/best_model.pth

# 5. Train different model with same structure:
python train.py --model maht-net --experiment-name baseline_v1
```

## Google Colab Integration

The same code works on Colab - just:
1. Mount Google Drive
2. Navigate to project directory
3. Run: `python train.py --model unet`

See `notebooks/train_unet_colab.ipynb` for full Colab example.
