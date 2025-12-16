# Experiment Results

This directory stores all training runs and their outputs, organized by model type.

## Directory Structure

```
experiments/results/
├── unet/
│   ├── experiment_name_20251216_104526/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pth
│   │   │   ├── last_model.pth
│   │   │   └── checkpoint_epoch_10.pth
│   │   ├── logs/
│   │   │   └── training.log
│   │   ├── config.json
│   │   └── metrics.json
│   └── another_experiment_20251217_093045/
│       └── ...
├── maht-net/
│   └── experiment_name_timestamp/
│       └── ...
├── resnet-keypoint/
│   └── experiment_name_timestamp/
│       └── ...
└── keypoint-rcnn/
    └── experiment_name_timestamp/
        └── ...
```

## Naming Convention

Each experiment run is saved with the following structure:
```
{model_name}/{experiment_name}_{timestamp}/
```

- **model_name**: Model type (unet, maht-net, resnet-keypoint, keypoint-rcnn)
- **experiment_name**: User-defined experiment name (or defaults to `{model}_training`)
- **timestamp**: Automatic timestamp in format `YYYYMMDD_HHMMSS`

## Experiment Contents

Each experiment directory contains:

### Checkpoints
- `best_model.pth` - Model with best validation performance
- `last_model.pth` - Most recent model state
- `checkpoint_epoch_N.pth` - Periodic checkpoints (every 10 epochs by default)

### Logs
- `training.log` - Complete training logs with metrics per epoch

### Configuration
- `config.json` - Full configuration used for training

### Metrics
- `metrics.json` - Training and validation metrics history

## Running Experiments

### Quick Test
```bash
python train.py --model unet --epochs 2 --batch-size 4 --experiment-name quick_test
```

### Full Training
```bash
python train.py --model unet --epochs 50 --batch-size 16 --experiment-name production_v1
```

### Resume Training
```bash
python train.py --model unet --resume experiments/results/unet/production_v1_20251216_104526/checkpoints/last_model.pth
```

## Finding Best Models

To find the best model for each architecture:
```bash
# List all UNet experiments sorted by date
ls -lt experiments/results/unet/

# Check metrics for a specific run
cat experiments/results/unet/production_v1_20251216_104526/metrics.json
```

## Cleanup

Remove old/failed experiments:
```bash
# Remove a specific experiment
rm -rf experiments/results/unet/failed_experiment_20251215_120000/

# Remove all experiments older than 30 days
find experiments/results -type d -mtime +30 -exec rm -rf {} +
```

## Tips

1. **Use descriptive experiment names** for easy identification
2. **Check metrics.json** to compare different runs
3. **Keep best models** and remove intermediate checkpoints to save space
4. **Document hyperparameters** in your experiment name (e.g., `unet_lr001_bs16`)
