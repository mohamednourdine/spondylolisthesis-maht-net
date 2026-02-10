# Step 3: Training Pipeline

## Overview

This guide covers training MAHT-Net on **Mac M3 Pro with MPS** (Metal Performance Shaders) backend. Your MacBook Pro is well-suited for local training with its 18GB unified memory and 18-core GPU.

---

## Training Configuration

### Your Hardware

| Component | Specification |
|-----------|---------------|
| Chip | Apple M3 Pro |
| CPU | 12-core (6P + 6E) |
| GPU | 18-core Apple GPU |
| Memory | 18 GB unified |
| Backend | MPS (PyTorch) |

### MPS vs CUDA

| Feature | MPS (Your Mac) | CUDA (Colab T4) |
|---------|----------------|-----------------|
| Memory | 18GB unified | 15GB dedicated |
| Mixed Precision | Limited support | Full AMP |
| Batch Size | 4-8 | 8-16 |
| Training Speed | ~80% of CUDA | Baseline |

### Training Hyperparameters

```python
# config/maht_net_config.py

class MAHTNetConfig:
    # Model
    NUM_KEYPOINTS_AP = 20
    NUM_KEYPOINTS_LA = 22
    D_MODEL = 256
    TRANSFORMER_LAYERS = 4
    VAM_LAYERS = 3
    
    # Training (optimized for M3 Pro)
    BATCH_SIZE = 4         # Conservative for 18GB unified memory
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4   # Lower than typical due to pretrained backbone
    WEIGHT_DECAY = 1e-4
    
    # LR Schedule
    LR_SCHEDULER = 'cosine'
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Loss weights
    HEATMAP_LOSS_WEIGHT = 1.0
    OFFSET_LOSS_WEIGHT = 0.1
    ANATOMICAL_LOSS_WEIGHT = 0.05
    
    # Data
    IMAGE_SIZE = (512, 512)
    HEATMAP_SIGMA = 4.0
    
    # Augmentation
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_RANGE = 10  # degrees
    SCALE_RANGE = (0.9, 1.1)
    
    # Training
    EARLY_STOPPING_PATIENCE = 20
    CHECKPOINT_EVERY = 10
    
    # Mixed precision (limited on MPS)
    USE_AMP = False  # MPS has limited AMP support, disable for stability
```

---

## Step 3.1: Loss Function Implementation

Create `training/maht_net_loss.py`:

```python
"""
Combined loss function for MAHT-Net training.

Components:
1. Heatmap Loss (focal loss for keypoint detection)
2. Offset Loss (L1 for sub-pixel refinement)
3. Anatomical Consistency Loss (structural priors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalHeatmapLoss(nn.Module):
    """
    CornerNet-style focal loss for heatmap regression.
    
    Reduces penalty for predictions near ground-truth keypoints
    and focuses training on hard negative examples.
    
    Loss = sum[(1-p_pred)^α * log(p_pred)]  at positive locations
         + sum[(1-p_gt)^β * p_pred^α * log(1-p_pred)]  at negative locations
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self, 
        pred_heatmaps: torch.Tensor, 
        gt_heatmaps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: (B, K, H, W) predicted heatmaps (logits)
            gt_heatmaps: (B, K, H, W) ground truth Gaussian heatmaps
            
        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred_heatmaps)
        
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-6, max=1-1e-6)
        
        # Positive locations (keypoint centers)
        pos_mask = gt_heatmaps.eq(1).float()
        neg_mask = gt_heatmaps.lt(1).float()
        
        # Number of keypoints (for normalization)
        num_pos = pos_mask.sum() + 1e-6
        
        # Positive loss: focus on accurate detection
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask
        
        # Negative loss: reduced penalty near keypoints
        neg_weight = torch.pow(1 - gt_heatmaps, self.beta)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask
        
        # Total loss (normalized by number of keypoints)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss


class AnatomicalConsistencyLoss(nn.Module):
    """
    Enforces anatomical constraints on keypoint predictions.
    
    Constraints:
    1. Vertical ordering: L1 should be above L2, L2 above L3, etc.
    2. Horizontal symmetry: Left-right corners should be roughly symmetric
    3. Vertebra shape: 4 corners should form valid quadrilateral
    """
    
    def __init__(self, margin: float = 5.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        pred_keypoints: torch.Tensor,
        view: str = 'AP'
    ) -> torch.Tensor:
        """
        Args:
            pred_keypoints: (B, K, 2) predicted keypoint coordinates
            view: 'AP' or 'LA'
            
        Returns:
            Anatomical consistency loss
        """
        B, K, _ = pred_keypoints.shape
        total_loss = 0.0
        
        # 1. Vertical ordering loss
        # Each row should be below the previous
        for i in range(0, K-2, 2):
            # Compare y-coordinates of consecutive rows
            y_current = pred_keypoints[:, i:i+2, 1].mean(dim=1)  # avg y of current row
            y_next = pred_keypoints[:, i+2:i+4, 1].mean(dim=1)   # avg y of next row
            
            # Next row should have larger y (lower in image)
            # ReLU(margin - (y_next - y_current))
            ordering_violation = F.relu(self.margin - (y_next - y_current))
            total_loss = total_loss + ordering_violation.mean()
        
        # 2. Horizontal symmetry (for AP view)
        if view == 'AP':
            for i in range(0, K, 2):
                x_left = pred_keypoints[:, i, 0]
                x_right = pred_keypoints[:, i+1, 0]
                
                # Right should be to the right of left
                symmetry_violation = F.relu(x_left - x_right + self.margin)
                total_loss = total_loss + symmetry_violation.mean()
        
        return total_loss / (K // 2)


class MAHTNetLoss(nn.Module):
    """
    Combined loss for MAHT-Net training.
    
    L_total = λ₁ * L_heatmap + λ₂ * L_offset + λ₃ * L_anatomical
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 0.1,
        anatomical_weight: float = 0.05,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0
    ):
        super().__init__()
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.anatomical_weight = anatomical_weight
        
        self.heatmap_loss = FocalHeatmapLoss(focal_alpha, focal_beta)
        self.offset_loss = nn.L1Loss()
        self.anatomical_loss = AnatomicalConsistencyLoss()
    
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        pred_keypoints: torch.Tensor = None,
        gt_keypoints: torch.Tensor = None,
        view: str = 'AP'
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            dict with 'total', 'heatmap', 'offset', 'anatomical' losses
        """
        losses = {}
        
        # Heatmap loss
        losses['heatmap'] = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
        
        # Offset loss (if keypoints provided)
        if pred_keypoints is not None and gt_keypoints is not None:
            losses['offset'] = self.offset_loss(pred_keypoints, gt_keypoints)
        else:
            losses['offset'] = torch.tensor(0.0, device=pred_heatmaps.device)
        
        # Anatomical consistency loss
        if pred_keypoints is not None:
            losses['anatomical'] = self.anatomical_loss(pred_keypoints, view)
        else:
            losses['anatomical'] = torch.tensor(0.0, device=pred_heatmaps.device)
        
        # Total loss
        losses['total'] = (
            self.heatmap_weight * losses['heatmap'] +
            self.offset_weight * losses['offset'] +
            self.anatomical_weight * losses['anatomical']
        )
        
        return losses
```

---

## Step 3.2: Training Script

Create `train_maht_net.py`:

```python
#!/usr/bin/env python3
"""
Training script for MAHT-Net on BUU-LSPINE dataset.
Optimized for Mac M3 Pro with MPS backend.
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.maht_net import create_maht_net
from src.data.buu_lspine_dataset import BUULSpineDataset
from training.maht_net_loss import MAHTNetLoss
from evaluation.metrics import compute_med


class MAHTNetTrainer:
    """Trainer for MAHT-Net model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        experiment_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Loss function
        self.criterion = MAHTNetLoss(
            heatmap_weight=config.get('heatmap_loss_weight', 1.0),
            offset_weight=config.get('offset_loss_weight', 0.1),
            anatomical_weight=config.get('anatomical_loss_weight', 0.05)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # LR Scheduler
        if config.get('lr_scheduler') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config['num_epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Best validation score
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_med': [], 'lr': []}
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'heatmap': 0, 'anatomical': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            gt_keypoints = batch['keypoints'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    pred_keypoints = self.model.extract_keypoints(outputs['heatmaps'])
                    
                    losses = self.criterion(
                        outputs['heatmaps'], gt_heatmaps,
                        pred_keypoints, gt_keypoints,
                        view=self.config.get('view', 'AP')
                    )
                
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                pred_keypoints = self.model.extract_keypoints(outputs['heatmaps'])
                
                losses = self.criterion(
                    outputs['heatmaps'], gt_heatmaps,
                    pred_keypoints, gt_keypoints,
                    view=self.config.get('view', 'AP')
                )
                
                losses['total'].backward()
                self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            pbar.set_postfix({'loss': losses['total'].item():.4f})
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model."""
        self.model.eval()
        val_losses = {'total': 0, 'heatmap': 0}
        all_preds = []
        all_gts = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            gt_keypoints = batch['keypoints'].to(self.device)
            
            outputs = self.model(images)
            pred_keypoints = self.model.extract_keypoints(outputs['heatmaps'])
            
            losses = self.criterion(outputs['heatmaps'], gt_heatmaps)
            
            val_losses['total'] += losses['total'].item()
            val_losses['heatmap'] += losses['heatmap'].item()
            
            # Collect for MED calculation
            all_preds.append(pred_keypoints.cpu())
            all_gts.append(gt_keypoints.cpu())
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute MED (Mean Euclidean Distance)
        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        med = compute_med(all_preds, all_gts)
        val_losses['med'] = med
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_model.pth')
            print(f'  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})')
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print("Starting MAHT-Net Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total'])
            else:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_med'].append(val_losses['med'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val MED: {val_losses['med']:.2f} pixels")
            print(f"  LR: {self.history['lr'][-1]:.6f}")
            
            # Check for best model
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save final history
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train MAHT-Net')
    parser.add_argument('--view', type=str, default='AP', choices=['AP', 'LA'])
    parser.add_argument('--data-dir', type=str, default='data/buu-lspine')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--experiment-name', type=str, default=None)
    args = parser.parse_args()
    
    # Device (MPS for Mac, CUDA for NVIDIA, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Config
    config = {
        'view': args.view,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine',
        'min_lr': 1e-6,
        'use_amp': False,  # Disable AMP for MPS stability
        'early_stopping_patience': 20,
        'heatmap_loss_weight': 1.0,
        'offset_loss_weight': 0.1,
        'anatomical_loss_weight': 0.05
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f'maht_net_{args.view}_{timestamp}'
    experiment_dir = Path('experiments/results') / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(experiment_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    num_keypoints = 20 if args.view == 'AP' else 22
    
    train_dataset = BUULSpineDataset(args.data_dir, view=args.view, split='train')
    val_dataset = BUULSpineDataset(args.data_dir, view=args.view, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model
    model = create_maht_net(view=args.view)
    
    # Train
    trainer = MAHTNetTrainer(
        model, train_loader, val_loader, config, device, experiment_dir
    )
    trainer.train()


if __name__ == '__main__':
    main()
```

---

## Step 3.3: MPS Backend Setup

Before training, verify your Mac's MPS backend:

```python
#!/usr/bin/env python3
"""Test MPS availability and performance."""

import torch
import time

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    
    # Quick benchmark
    x = torch.randn(32, 3, 512, 512, device=device)
    
    # Warmup
    for _ in range(10):
        _ = x.sum()
    torch.mps.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        y = x * 2 + 1
    torch.mps.synchronize()
    elapsed = time.time() - start
    
    print(f"MPS benchmark: {elapsed:.3f}s for 100 iterations")
    print("MPS is working correctly!")
```

### Common MPS Issues & Solutions

| Issue | Solution |
|-------|----------|
| "MPS backend not available" | Update to macOS 12.3+, PyTorch 1.12+ |
| Memory errors | Reduce batch size from 4 to 2 |
| Slow training | Ensure no apps consuming GPU |
| NaN losses | Disable AMP, use float32 |

---

## Step 3.4: Training Tips for Mac M3 Pro

### Memory Optimization

```python
# For 18GB unified memory, recommended settings:
BATCH_SIZE = 4           # Safe starting point
NUM_WORKERS = 4          # Don't exceed CPU cores / 2
PREFETCH_FACTOR = 2      # Default is fine

# Monitor memory with Activity Monitor or:
import os
print(f"Memory usage: {os.popen('vm_stat').read()}")

# Clear MPS cache if needed
torch.mps.empty_cache()
```

### Background Process Management

```bash
# Before training, close heavy apps (Chrome, Docker, etc.)
# Check what's using memory:
top -l 1 | head -20
```

### Resume Training After Interruption

```python
# Resume from checkpoint
checkpoint = torch.load('experiments/results/maht_net_ap/latest_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Mac-Specific Tips

1. **Keep Mac plugged in** during training
2. **Disable sleep**: System Settings → Energy → Prevent sleep when display is off
3. **Monitor temperature**: `sudo powermetrics --samplers smc` (shows thermal throttling)
4. **Use tmux/screen** to keep training running if terminal closes

---

## Training Commands Summary

```bash
# Quick test run (verify everything works)
python train_maht_net.py --view AP --batch-size 2 --epochs 2

# Full AP training (Mac M3 Pro with MPS)
python train_maht_net.py \
    --view AP \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-4

# Train LA view
python train_maht_net.py \
    --view LA \
    --batch-size 4 \
    --epochs 100

# Resume training
python train_maht_net.py --resume experiments/results/maht_net_ap/latest_checkpoint.pth

# Background training with nohup (keeps running after terminal closes)
nohup python train_maht_net.py --view AP --batch-size 4 --epochs 100 > training_ap.log 2>&1 &
```

---

## Expected Training Timeline (Mac M3 Pro)

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup | 15 min | Verify MPS, test dataloader |
| Initial epochs (1-10) | ~3 hours | Rapid loss decrease |
| Middle (11-50) | ~15 hours | Gradual improvement |
| Final (51-100) | ~15 hours | Fine-tuning, possible early stop |
| **Total** | ~33 hours | Can run overnight |

**With batch_size=4**: ~3 min/epoch (2520 train + 540 val images)

**Tip**: Use `tmux` or `screen` to keep training running in background.

---

## Next Step

After training, proceed to:
- [04_evaluation_protocol.md](04_evaluation_protocol.md) - Evaluate model performance

---

*Last Updated: February 2025*
