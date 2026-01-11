# Quick Reference Guide

## 🚀 Common Commands

### Testing (Local PC - Small Dataset)
```bash
# Quick validation (RECOMMENDED before cloud training)
python tests/test_training_small.py

# Test specific components
python tests/test_unet.py
python tests/test_global_evaluator.py
```

### Training

#### Local Testing
```bash
# Quick test with 2 epochs, batch size 2
python train.py --model unet --epochs 2 --batch-size 2 --experiment-name quick_test
```

#### Cloud Training (Google Colab / Full Dataset)
```bash
# Production training (batch size 8 for T4 GPU 15GB)
python train.py --model unet --epochs 50 --batch-size 8 --experiment-name production_v1

# With custom learning rate
python train.py --model unet --epochs 50 --batch-size 8 --lr 0.0005 --experiment-name lr_0005

# For GPUs with more memory (V100, A100), you can use larger batch sizes:
python train.py --model unet --epochs 50 --batch-size 16 --experiment-name production_v1
```

#### Resume Training
```bash
python train.py --model unet \
  --resume experiments/results/unet/production_v1_20251216_104526/checkpoints/last_model.pth
```

### Experiment Management

#### Compare Results
```bash
# Compare all UNet experiments
python scripts/compare_experiments.py --model unet

# Show top 5 best experiments
python scripts/compare_experiments.py --model unet --top 5

# Compare across all models
python scripts/compare_experiments.py --list-all

# Detailed metrics for specific experiment
python scripts/compare_experiments.py --model unet --details "production_v1"
```

#### Find Best Model
```bash
# List experiments sorted by validation loss
python scripts/compare_experiments.py --model unet

# Check best model path
ls -t experiments/results/unet/*/checkpoints/best_model.pth | head -1
```

## 📂 Project Structure Quick Reference

```
.
├── train.py                    # Main training script
├── tests/                      # All test scripts
│   └── test_training_small.py  # Quick validation (USE THIS FIRST)
├── experiments/results/        # Training outputs
│   └── {model}/                # Organized by model type
│       └── {name}_{timestamp}/ # Each experiment run
│           ├── checkpoints/
│           │   ├── best_model.pth
│           │   └── last_model.pth
│           ├── config.json
│           └── training_history.json
├── config/                     # Model configurations
├── models/                     # Model architectures
├── src/data/                   # Datasets & augmentation
├── evaluation/                 # Metrics & evaluators
└── scripts/                    # Utility scripts
```

## 📊 Metrics Tracked During Training

- **Loss**: Training and validation loss (focal loss for heatmaps)
- **MRE**: Mean Radial Error (pixels) - lower is better
- **SDR**: Successful Detection Rate at thresholds:
  - 2.0mm, 2.5mm, 3.0mm, 4.0mm - higher is better

Example output:
```
Epoch [25/50] - 120.5s
  Train Loss: -0.4567
  Val Loss:   -0.3891
  Val MRE:    28.45
  Val SDR:    2.0mm: 0.7234, 2.5mm: 0.8156, 3.0mm: 0.8891, 4.0mm: 0.9423
  ✓ New best model (val_loss: -0.3891)
```

## 🔧 Configuration

### Override Config via Command Line
```bash
# Change batch size
python train.py --model unet --batch-size 32

# Change learning rate
python train.py --model unet --lr 0.0001

# Change number of epochs
python train.py --model unet --epochs 100
```

### Custom Config File
```bash
# Use custom YAML config
python train.py --model unet --config path/to/custom_config.yaml

# Use custom Python config
python train.py --model unet --config config/custom_unet_config.py
```

## 🌐 Google Colab Setup

1. **Upload project to Colab**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/spondylolisthesis-maht-net
   ```

2. **Install dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```

3. **Run training**:
   ```bash
   !python train.py --model unet --epochs 50 --batch-size 8 --experiment-name colab_run1
   ```
   
   **Note**: Batch size 8 is recommended for T4 GPUs (15GB). Use `--batch-size 16` for V100/A100.

4. **Monitor training**:
   ```python
   # Training metrics are displayed in real-time
   # Check results after training
   !python scripts/compare_experiments.py --model unet
   ```

## 🎯 Workflow

1. **Test Locally** (with small data):
   ```bash
   python tests/test_training_small.py
   ```

2. **Train in Cloud** (full dataset):
   ```bash
   python train.py --model unet --epochs 50 --batch-size 8 --experiment-name production_v1
   ```

3. **Compare Results**:
   ```bash
   python scripts/compare_experiments.py --model unet
   ```

4. **Use Best Model**:
   ```bash
   # Best model saved at:
   # experiments/results/unet/{experiment_name}/checkpoints/best_model.pth
   ```

## ⚡ Tips

- Use `test_training_small.py` before any full training run
- Descriptive experiment names help: `unet_lr001_bs8_focal`
- Check `training_history.json` for detailed metrics
- Best models are automatically saved during training
- Resume from `last_model.pth` if training is interrupted
- **Batch size 8** is optimal for T4 GPUs (15GB memory)
- Use `--batch-size 4` if you still get OOM errors
- Clear CUDA cache is automatically handled between batches
