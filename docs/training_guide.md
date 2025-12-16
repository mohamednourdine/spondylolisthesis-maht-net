# Complete Training Guide: Establishing Baselines on Spondylolisthesis Dataset

**Project**: Automated Spondylolisthesis Grading Using Deep Learning  
**Goal**: Train 4 baseline models and establish benchmark performance  
**Timeline**: 6-8 weeks from data preparation to publication-ready results

---

## 📋 Table of Contents

1. [Overview & Strategy](#overview--strategy)
2. [Pre-Training Checklist](#pre-training-checklist)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: Baseline Models Training](#phase-2-baseline-models-training)
5. [Phase 3: Evaluation & Analysis](#phase-3-evaluation--analysis)
6. [Phase 4: Results Compilation](#phase-4-results-compilation)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Quick Reference Commands](#quick-reference-commands)

---

## Overview & Strategy

### Training Order (Recommended)

Train models from **simplest to most complex** to build confidence and debug issues early:

```
Week 1-2: U-Net (Simplest)
    ↓
Week 3: ResNet Keypoint Detector
    ↓
Week 4-5: Keypoint R-CNN
    ↓
Week 6-7: MAHT-Net (Most Complex)
    ↓
Week 8: Comparison & Paper Writing
```

### Success Metrics

Each model should achieve:
- **Training Loss**: Converges smoothly (no NaN, no divergence)
- **Validation MRE**: < 5mm (target: 2-3mm for best model)
- **SDR@2mm**: > 70% (target: 85%+ for best model)
- **Clinical Grade Accuracy**: > 80% (target: 90%+ for best model)

---

## Pre-Training Checklist

### ✅ Before Starting ANY Training

- [ ] **Environment Setup**
  ```bash
  conda activate phd
  python --version  # Should be 3.9.x
  python -c "import torch; print(torch.__version__)"  # Should be 1.13.0
  python -c "import torch; print(torch.cuda.is_available())"  # Check GPU
  ```

- [ ] **Data Verification**
  ```bash
  # Check data directory structure
  ls -la data/Train/images/ | head -5
  ls -la data/Train/labels/ | head -5
  ls -la data/Validation/images/ | head -5
  ls -la data/Validation/labels/ | head -5
  
  # Count files
  echo "Training images: $(ls data/Train/images/*.jpg | wc -l)"
  echo "Training labels: $(ls data/Train/labels/*.json | wc -l)"
  echo "Validation images: $(ls data/Validation/images/*.jpg | wc -l)"
  echo "Validation labels: $(ls data/Validation/labels/*.json | wc -l)"
  ```

- [ ] **Run Data Understanding Notebook**
  ```bash
  jupyter notebook notebooks/00_data_understanding.ipynb
  # Execute all cells, verify no errors
  ```

- [ ] **Run Preprocessing Pipeline**
  ```bash
  jupyter notebook notebooks/01_preprocessing_pipeline.ipynb
  # Execute all cells, verify augmentations work
  ```

- [ ] **Create Experiment Tracking Directory**
  ```bash
  mkdir -p experiments/results/{unet,resnet,keypoint_rcnn,maht_net}
  mkdir -p experiments/checkpoints/{unet,resnet,keypoint_rcnn,maht_net}
  mkdir -p experiments/logs/{unet,resnet,keypoint_rcnn,maht_net}
  ```

---

## Phase 1: Data Preparation

### Step 1.1: Verify Dataset Structure

**Expected Structure:**
```
data/
├── Train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ... (494 total)
│   └── labels/
│       ├── image_001.json
│       ├── image_002.json
│       └── ... (494 total)
├── Validation/
│   ├── images/ (206 images)
│   └── labels/ (206 labels)
└── Clinical/
    ├── images/ (16 images)
    └── labels/ (16 labels)
```

**Verification Script:**
```python
# Create: scripts/verify_data.py

import os
import json
from pathlib import Path

def verify_dataset():
    """Verify dataset integrity before training."""
    
    data_root = Path("data")
    splits = ["Train", "Validation", "Clinical"]
    issues = []
    
    for split in splits:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        
        if not img_dir.exists():
            issues.append(f"Missing: {img_dir}")
            continue
            
        images = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.json"))
        
        print(f"\n{split}:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check matching files
        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}
        
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} images without labels")
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} labels without images")
        
        # Validate JSON format
        for lbl_file in list(labels)[:5]:  # Check first 5
            with open(lbl_file) as f:
                data = json.load(f)
                required = ["boxes", "keypoints", "labels"]
                if not all(k in data for k in required):
                    issues.append(f"Invalid JSON structure in {lbl_file.name}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Dataset verification PASSED!")
        return True

if __name__ == "__main__":
    verify_dataset()
```

**Run Verification:**
```bash
python scripts/verify_data.py
```

### Step 1.2: Calculate Dataset Statistics

**Create: `scripts/calculate_stats.py`**

```python
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def calculate_image_stats():
    """Calculate mean and std for normalization."""
    
    img_dir = Path("data/Train/images")
    images = list(img_dir.glob("*.jpg"))
    
    print(f"Calculating stats from {len(images)} training images...")
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0
    
    for img_path in tqdm(images):
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        pixel_sum += img_array.sum(axis=(0, 1))
        pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
        num_pixels += img_array.shape[0] * img_array.shape[1]
    
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"\nAdd these to your config files!")
    
    return mean, std

if __name__ == "__main__":
    calculate_image_stats()
```

**Run:**
```bash
python scripts/calculate_stats.py
```

---

## Phase 2: Baseline Models Training

### Model 1: U-Net (Week 1-2)

**Why Start Here?**
- Simplest architecture
- Fast training (1-2 days)
- Good for debugging data pipeline
- Establishes lower baseline

#### Configuration: `experiments/configs/unet_config.yaml`

```yaml
model:
  name: unet
  architecture:
    encoder_channels: [64, 128, 256, 512]
    decoder_channels: [512, 256, 128, 64]
    num_keypoints: 16  # 4 vertebrae × 4 corners
    
training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  scheduler: 
    type: reduce_on_plateau
    patience: 10
    factor: 0.5
  early_stopping:
    patience: 20
    
data:
  image_size: [512, 512]
  normalize:
    mean: [0.485, 0.456, 0.406]  # Update with calculated stats
    std: [0.229, 0.224, 0.225]   # Update with calculated stats
  augmentation:
    horizontal_flip: 0.5
    rotation: 10
    brightness: 0.2
    contrast: 0.2
    
paths:
  train_images: data/Train/images
  train_labels: data/Train/labels
  val_images: data/Validation/images
  val_labels: data/Validation/labels
  checkpoint_dir: experiments/checkpoints/unet
  log_dir: experiments/logs/unet
  
seed: 42
```

#### Training Script: `scripts/train_unet.py`

```python
#!/usr/bin/env python3
"""
Train U-Net for vertebral landmark detection.

Usage:
    python scripts/train_unet.py --config experiments/configs/unet_config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from models.unet import UNet
from src.data.dataset import SpondylolisthesisDataset
from training.losses import KeypointMSELoss
from evaluation.metrics import calculate_mre, calculate_sdr
from utils.helpers import set_seed, save_checkpoint, load_checkpoint

def setup_logging(log_dir):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        keypoints = targets['keypoints'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, keypoints)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch."""
    model.eval()
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, targets in pbar:
            images = images.to(device)
            keypoints = targets['keypoints'].to(device)
            
            predictions = model(images)
            loss = criterion(predictions, keypoints)
            
            epoch_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(keypoints.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / len(dataloader)
    
    # Calculate metrics
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mre = calculate_mre(predictions, targets)
    sdr_2mm = calculate_sdr(predictions, targets, threshold=2.0)
    sdr_4mm = calculate_sdr(predictions, targets, threshold=4.0)
    
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"MRE: {mre:.2f}mm | SDR@2mm: {sdr_2mm:.2f}% | SDR@4mm: {sdr_4mm:.2f}%")
    
    return avg_loss, mre, sdr_2mm, sdr_4mm

def main(config_path):
    """Main training function."""
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_seed(config['seed'])
    logger = setup_logging(config['paths']['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SpondylolisthesisDataset(
        image_dir=config['paths']['train_images'],
        label_dir=config['paths']['train_labels'],
        transform=None,  # Add augmentation
        image_size=tuple(config['data']['image_size'])
    )
    
    val_dataset = SpondylolisthesisDataset(
        image_dir=config['paths']['val_images'],
        label_dir=config['paths']['val_labels'],
        transform=None,
        image_size=tuple(config['data']['image_size'])
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = UNet(
        in_channels=3,
        num_keypoints=config['model']['architecture']['num_keypoints']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = KeypointMSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor']
    )
    
    # Training loop
    best_mre = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_loss, mre, sdr_2mm, sdr_4mm = validate_epoch(
            model, val_loader, criterion, device, logger
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if mre < best_mre:
            best_mre = mre
            patience_counter = 0
            
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
            save_checkpoint(
                model, optimizer, epoch, mre, checkpoint_path
            )
            logger.info(f"✅ New best model saved! MRE: {mre:.2f}mm")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"Best MRE: {best_mre:.2f}mm")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
```

#### Run Training:

```bash
# Activate environment
conda activate phd

# Start training (takes 1-2 days on GPU, 1 week on CPU)
python scripts/train_unet.py --config experiments/configs/unet_config.yaml

# Monitor in real-time (open new terminal)
tail -f experiments/logs/unet/training.log
```

#### Expected Output:

```
================================================================
Epoch 1/100
================================================================
Training: 100%|████████████| 62/62 [01:23<00:00, loss=0.0245]
Training Loss: 0.0245
Validation: 100%|██████████| 26/26 [00:15<00:00, loss=0.0198]
Validation Loss: 0.0198
MRE: 4.52mm | SDR@2mm: 65.32% | SDR@4mm: 85.71%
Learning Rate: 0.001000
✅ New best model saved! MRE: 4.52mm
================================================================
```

#### Success Criteria:

- [ ] Training completes without errors
- [ ] Loss decreases smoothly (no NaN or inf)
- [ ] Validation MRE < 5mm by epoch 50
- [ ] Best model saved correctly
- [ ] Can visualize predictions on validation set

---

### Model 2: ResNet Keypoint Detector (Week 3)

Similar structure but update config:

```yaml
# experiments/configs/resnet_config.yaml
model:
  name: resnet_keypoint
  backbone: resnet50
  pretrained: true
  num_keypoints: 16
  
training:
  batch_size: 16  # Can use larger batch
  epochs: 80
  learning_rate: 0.0001  # Lower LR for pretrained model
```

**Training Command:**
```bash
python scripts/train_resnet.py --config experiments/configs/resnet_config.yaml
```

---

### Model 3: Keypoint R-CNN (Week 4-5)

```yaml
# experiments/configs/keypoint_rcnn_config.yaml
model:
  name: keypoint_rcnn
  backbone: resnet50_fpn
  pretrained: true
  num_keypoints: 16
  box_detections_per_img: 4  # 4 vertebrae
  
training:
  batch_size: 4  # Smaller batch for R-CNN
  epochs: 100
  learning_rate: 0.0005
```

**Training Command:**
```bash
python scripts/train_keypoint_rcnn.py --config experiments/configs/keypoint_rcnn_config.yaml
```

---

### Model 4: MAHT-Net (Week 6-7)

```yaml
# experiments/configs/maht_net_config.yaml
model:
  name: maht_net
  # MAHT-Net specific parameters
  num_keypoints: 16
  attention_heads: 8
  transformer_layers: 6
  
training:
  batch_size: 8
  epochs: 120
  learning_rate: 0.0001
```

**Training Command:**
```bash
python scripts/train_maht_net.py --config experiments/configs/maht_net_config.yaml
```

---

## Phase 3: Evaluation & Analysis

### Comprehensive Evaluation Script

**Create: `scripts/evaluate_all_models.py`**

```python
#!/usr/bin/env python3
"""
Evaluate all trained models and generate comparison report.

Usage:
    python scripts/evaluate_all_models.py
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import (
    calculate_mre, 
    calculate_sdr,
    calculate_slip_percentage,
    meyerding_grade_accuracy
)

def evaluate_model(model_name, checkpoint_path, test_loader, device):
    """Evaluate a single model."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model = load_model(model_name, checkpoint_path)
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            predictions = model(images)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets['keypoints'].cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate all metrics
    results = {
        'Model': model_name,
        'MRE (mm)': calculate_mre(predictions, targets),
        'SDR@2mm (%)': calculate_sdr(predictions, targets, 2.0),
        'SDR@4mm (%)': calculate_sdr(predictions, targets, 4.0),
        'Slip MAE (%)': calculate_slip_percentage(predictions, targets),
        'Grade Accuracy (%)': meyerding_grade_accuracy(predictions, targets)
    }
    
    return results, predictions, targets

def create_comparison_table(all_results):
    """Create comparison table of all models."""
    
    df = pd.DataFrame(all_results)
    df = df.round(2)
    
    # Save to CSV
    df.to_csv('experiments/results/model_comparison.csv', index=False)
    
    # Create formatted table
    print("\n" + "="*80)
    print("MODEL COMPARISON - BASELINE RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Highlight best performers
    print("\n🏆 BEST PERFORMERS:")
    print(f"  Lowest MRE: {df.loc[df['MRE (mm)'].idxmin(), 'Model']} "
          f"({df['MRE (mm)'].min():.2f}mm)")
    print(f"  Highest SDR@2mm: {df.loc[df['SDR@2mm (%)'].idxmax(), 'Model']} "
          f"({df['SDR@2mm (%)'].max():.2f}%)")
    print(f"  Best Grade Accuracy: {df.loc[df['Grade Accuracy (%)'].idxmax(), 'Model']} "
          f"({df['Grade Accuracy (%)'].max():.2f}%)")
    
    return df

def create_visualization(df):
    """Create comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MRE comparison
    ax = axes[0, 0]
    df.plot(x='Model', y='MRE (mm)', kind='bar', ax=ax, color='skyblue', legend=False)
    ax.set_title('Mean Radial Error (MRE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MRE (mm)')
    ax.axhline(y=2.0, color='r', linestyle='--', label='Target: 2mm')
    ax.legend()
    
    # SDR comparison
    ax = axes[0, 1]
    df.plot(x='Model', y='SDR@2mm (%)', kind='bar', ax=ax, color='lightgreen', legend=False)
    ax.set_title('Success Detection Rate @ 2mm', fontsize=14, fontweight='bold')
    ax.set_ylabel('SDR (%)')
    ax.axhline(y=85, color='r', linestyle='--', label='Target: 85%')
    ax.legend()
    
    # Slip MAE
    ax = axes[1, 0]
    df.plot(x='Model', y='Slip MAE (%)', kind='bar', ax=ax, color='coral', legend=False)
    ax.set_title('Slip Percentage Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (%)')
    
    # Grade Accuracy
    ax = axes[1, 1]
    df.plot(x='Model', y='Grade Accuracy (%)', kind='bar', ax=ax, color='gold', legend=False)
    ax.set_title('Meyerding Grade Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.axhline(y=90, color='r', linestyle='--', label='Target: 90%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Comparison plot saved: experiments/results/model_comparison.png")

def main():
    """Main evaluation function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define models to evaluate
    models = {
        'U-Net': 'experiments/checkpoints/unet/best_model.pth',
        'ResNet': 'experiments/checkpoints/resnet/best_model.pth',
        'Keypoint R-CNN': 'experiments/checkpoints/keypoint_rcnn/best_model.pth',
        'MAHT-Net': 'experiments/checkpoints/maht_net/best_model.pth'
    }
    
    # Load test dataset
    test_loader = create_test_loader()
    
    # Evaluate each model
    all_results = []
    for model_name, checkpoint_path in models.items():
        if Path(checkpoint_path).exists():
            results, _, _ = evaluate_model(model_name, checkpoint_path, test_loader, device)
            all_results.append(results)
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # Create comparison
    if all_results:
        df = create_comparison_table(all_results)
        create_visualization(df)
    else:
        print("❌ No models found to evaluate!")

if __name__ == "__main__":
    main()
```

**Run Evaluation:**
```bash
python scripts/evaluate_all_models.py
```

---

## Phase 4: Results Compilation

### Generate Publication-Ready Results

**Create: `scripts/generate_paper_results.py`**

```python
#!/usr/bin/env python3
"""
Generate all tables and figures for the paper.

Outputs:
    - Table 1: Model architectures comparison
    - Table 2: Landmark detection performance
    - Table 3: Clinical grading performance
    - Figure 1: Visual comparison of predictions
    - Figure 2: Error distribution analysis
    - Figure 3: Per-vertebra performance breakdown
"""

# Implementation details...
```

**Run:**
```bash
python scripts/generate_paper_results.py
```

**Outputs:**
- `experiments/results/table1_architectures.tex`
- `experiments/results/table2_landmark_performance.tex`
- `experiments/results/table3_clinical_performance.tex`
- `experiments/results/figure1_visual_comparison.png`
- `experiments/results/figure2_error_analysis.png`
- `experiments/results/figure3_per_vertebra.png`

---

## Troubleshooting Guide

### Common Issues

#### 1. Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size in config
batch_size: 4  # Instead of 8

# Or use gradient accumulation
python scripts/train_unet.py --config config.yaml --accumulation-steps 4

# Or use CPU (slower)
python scripts/train_unet.py --config config.yaml --device cpu
```

#### 2. NaN Loss

**Error:** Loss becomes NaN during training

**Solutions:**
```yaml
# In config, reduce learning rate
learning_rate: 0.0001  # Instead of 0.001

# Add gradient clipping
training:
  gradient_clip: 1.0
```

#### 3. No Convergence

**Issue:** Loss not decreasing after many epochs

**Solutions:**
- Check data normalization
- Verify labels are correct
- Try different learning rate
- Check for data loading bugs

```python
# Debug data loading
from src.data.dataset import SpondylolisthesisDataset

dataset = SpondylolisthesisDataset(...)
img, target = dataset[0]
print(f"Image range: [{img.min()}, {img.max()}]")
print(f"Keypoints: {target['keypoints']}")
```

---

## Quick Reference Commands

### Training

```bash
# U-Net
python scripts/train_unet.py --config experiments/configs/unet_config.yaml

# ResNet
python scripts/train_resnet.py --config experiments/configs/resnet_config.yaml

# Keypoint R-CNN
python scripts/train_keypoint_rcnn.py --config experiments/configs/keypoint_rcnn_config.yaml

# MAHT-Net
python scripts/train_maht_net.py --config experiments/configs/maht_net_config.yaml
```

### Monitoring

```bash
# Watch training logs
tail -f experiments/logs/unet/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Tensorboard (if implemented)
tensorboard --logdir experiments/logs
```

### Evaluation

```bash
# Evaluate all models
python scripts/evaluate_all_models.py

# Evaluate single model
python scripts/evaluate_model.py --model unet --checkpoint experiments/checkpoints/unet/best_model.pth

# Generate paper results
python scripts/generate_paper_results.py
```

### Resuming Training

```bash
# Resume from checkpoint
python scripts/train_unet.py --config config.yaml --resume experiments/checkpoints/unet/checkpoint_epoch50.pth
```

---

## Training Timeline Summary

| Week | Task | Deliverable | Status |
|------|------|-------------|--------|
| 1 | U-Net training | Baseline results | ⏳ |
| 2 | U-Net refinement | Best U-Net model | ⏳ |
| 3 | ResNet training | ResNet results | ⏳ |
| 4 | R-CNN training | R-CNN results | ⏳ |
| 5 | R-CNN refinement | Best R-CNN model | ⏳ |
| 6 | MAHT-Net training | MAHT-Net results | ⏳ |
| 7 | MAHT-Net refinement | Best MAHT-Net model | ⏳ |
| 8 | Comparison & paper | Publication draft | ⏳ |

---

## Success Checklist

### Before Submission

- [ ] All 4 models trained successfully
- [ ] Best checkpoints saved for each model
- [ ] Comprehensive evaluation completed
- [ ] Comparison table generated
- [ ] Visualization figures created
- [ ] Error analysis performed
- [ ] Results reproducible
- [ ] Code cleaned and commented
- [ ] README updated with results
- [ ] Paper draft written

---

## Next Steps After Training

1. **Write Paper** (2 weeks)
   - Introduction
   - Methods
   - Results
   - Discussion

2. **Code Release** (1 week)
   - Clean up code
   - Add documentation
   - Create GitHub release
   - Add pretrained models

3. **Submit to ArXiv** (1 day)
   - Upload paper
   - Share on social media
   - Email dataset authors

4. **Conference Submission** (ongoing)
   - Target: MICCAI, ISBI, or similar
   - Prepare 8-page paper
   - Create poster/slides

---

**Good luck with training! 🚀**

You're establishing the first baselines on this dataset - that's valuable research!
