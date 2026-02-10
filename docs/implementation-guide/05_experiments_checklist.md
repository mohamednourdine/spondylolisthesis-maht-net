# Step 5: Experiments Checklist

## Overview

This document outlines all experiments needed for the MAHT-Net paper, including ablation studies, baseline comparisons, and analysis.

---

## Required Experiments for Paper

### 📋 Main Experiments

| # | Experiment | Priority | Status | Notes |
|---|------------|----------|--------|-------|
| 1 | Train MAHT-Net on AP view | **Critical** | ⬜ | Primary result |
| 2 | Train MAHT-Net on LA view | **Critical** | ⬜ | Primary result |
| 3 | Evaluate on test set | **Critical** | ⬜ | Final metrics |
| 4 | Compare vs Klinwichit baseline | **Critical** | ⬜ | Published: 4.63mm AP, 4.91mm LA |

**Note**: Baseline comparison uses published Klinwichit et al. 2023 results (ResNet152V2: MED 4.63mm AP, 4.91mm LA) - no need to retrain baselines.

### 📊 Ablation Studies (VAM Focus)

| # | Study | Purpose | Status |
|---|-------|---------|--------|
| A1 | Without Transformer Bridge | Show global context value | ⬜ |
| A2 | Without VAM | Show anatomical attention value | ⬜ |
| A3 | Without DARK decoding | Show sub-pixel refinement value | ⬜ |
| A4 | Different VAM layer counts (1,2,3,4) | Architecture search | ⬜ |
| A5 | Different Transformer layers (2,4,6) | Architecture search | ⬜ |

**Note**: Focus ablations on VAM (our key contribution). Backbone choice (EfficientNetV2-S) justified by literature.

### 📈 Analysis

| # | Analysis | Purpose | Status |
|---|----------|---------|--------|
| B1 | Per-vertebra error analysis | Identify difficult cases | ⬜ |
| B2 | Error vs image quality | Robustness analysis | ⬜ |
| B3 | Failure case analysis | Understand limitations | ⬜ |
| B4 | Attention visualization | Interpretability | ⬜ |
| B5 | Inference time comparison | Efficiency | ⬜ |

---

## Experiment Configurations

### Main Experiments

```yaml
# experiments/configs/maht_net_ap.yaml
experiment_name: maht_net_ap_main
model: maht-net
view: AP
num_keypoints: 20

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: cosine
  warmup_epochs: 5
  early_stopping_patience: 20

model_config:
  backbone: efficientnet_v2_s
  pretrained: true
  freeze_backbone_stages: 2
  d_model: 256
  transformer_layers: 4
  vam_layers: 3
  dropout: 0.1

loss:
  heatmap_weight: 1.0
  offset_weight: 0.1
  anatomical_weight: 0.05
```

### Ablation Study Configs

```yaml
# A1: Without Transformer
experiment_name: ablation_no_transformer
model: maht-net
model_config:
  transformer_layers: 0  # Disable transformer

# A2: Without VAM
experiment_name: ablation_no_vam
model: maht-net
model_config:
  vam_layers: 0  # Disable VAM

# A3: Without DARK
experiment_name: ablation_no_dark
model: maht-net
evaluation:
  use_dark: false

# A4: VAM layers
# Run with vam_layers: 1, 2, 3, 4

# A5: Transformer layers
# Run with transformer_layers: 2, 4, 6
```

---

## Running Experiments

### Batch Experiment Runner

Create `scripts/run_experiments.py`:

```python
#!/usr/bin/env python3
"""Run all experiments for the paper."""

import subprocess
import yaml
from pathlib import Path
from datetime import datetime

EXPERIMENTS = {
    # Main experiments
    'main_ap': {
        'view': 'AP',
        'config': 'experiments/configs/maht_net_ap.yaml'
    },
    'main_la': {
        'view': 'LA', 
        'config': 'experiments/configs/maht_net_la.yaml'
    },
    
    # Ablations (VAM focus)
    'ablation_no_transformer': {
        'view': 'AP',
        'transformer_layers': 0
    },
    'ablation_no_vam': {
        'view': 'AP',
        'vam_layers': 0
    },
    'ablation_no_dark': {
        'view': 'AP',
        'use_dark': False
    },
}

def run_experiment(name: str, config: dict):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    cmd = ['python', 'train_maht_net.py']
    
    # Add config options
    for key, value in config.items():
        if key == 'config':
            cmd.extend(['--config', value])
        else:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    cmd.extend(['--experiment-name', name])
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    print(f"Completed: {name}")
    return True


def main():
    """Run all experiments."""
    results = {}
    
    for name, config in EXPERIMENTS.items():
        success = run_experiment(name, config)
        results[name] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == '__main__':
    main()
```

---

## Expected Results Table

### Main Results (Target)

| Model | View | MED (mm) | PCK@10 | Improvement vs Baseline |
|-------|------|----------|--------|-------------------------|
| Klinwichit et al. (ResNet152V2) | AP | 4.63 | - | Baseline |
| Klinwichit et al. (ResNet152V2) | LA | 4.91 | - | Baseline |
| **MAHT-Net (Ours)** | AP | **< 4.0** | **> 90%** | **> 13%** |
| **MAHT-Net (Ours)** | LA | **< 4.0** | **> 90%** | **> 18%** |

### Ablation Study Results (Template)

| Configuration | MED (mm) | Δ from Full Model |
|---------------|----------|-------------------|
| Full MAHT-Net | X.XX | - |
| w/o Transformer | X.XX | +X.XX |
| w/o VAM | X.XX | +X.XX |
| w/o DARK | X.XX | +X.XX |
| VAM 1 layer | X.XX | +X.XX |
| VAM 2 layers | X.XX | +X.XX |
| VAM 4 layers | X.XX | +X.XX |

---

## Plotting Scripts

### Create Results Figures

```python
#!/usr/bin/env python3
"""Generate figures for paper."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_ablation_comparison(results_dir: str, output_path: str):
    """Create ablation study bar chart."""
    
    # Load results
    experiments = [
        ('Full Model', 'maht_net_ap_main'),
        ('w/o Transformer', 'ablation_no_transformer'),
        ('w/o VAM', 'ablation_no_vam'),
        ('w/o DARK', 'ablation_no_dark'),
    ]
    
    meds = []
    names = []
    
    for name, exp_name in experiments:
        metrics_file = Path(results_dir) / exp_name / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                meds.append(metrics['med_mm'])
                names.append(name)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71'] + ['#e74c3c'] * (len(names) - 1)  # Green for full, red for ablations
    
    bars = ax.bar(names, meds, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add baseline line
    ax.axhline(y=4.63, color='gray', linestyle='--', label='Baseline (Klinwichit)')
    ax.axhline(y=4.0, color='green', linestyle=':', label='Target')
    
    ax.set_ylabel('Mean Euclidean Distance (mm)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.legend()
    
    # Add value labels
    for bar, med in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{med:.2f}', ha='center', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")


def plot_training_curves(experiment_dir: str, output_path: str):
    """Plot training curves."""
    
    with open(Path(experiment_dir) / 'training_history.json') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MED
    axes[1].plot(epochs, history['val_med'], 'g-')
    axes[1].axhline(y=4.0, color='red', linestyle='--', label='Target')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MED (pixels)')
    axes[1].set_title('Validation MED')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")


def plot_per_vertebra_analysis(metrics_file: str, output_path: str):
    """Plot per-vertebra error analysis."""
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    vertebrae = ['L1', 'L2', 'L3', 'L4', 'L5']
    meds = [metrics[f'med_{v}'] for v in vertebrae]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vertebrae)))
    
    bars = ax.bar(vertebrae, meds, color=colors, edgecolor='black')
    
    ax.set_xlabel('Vertebra')
    ax.set_ylabel('MED (pixels)')
    ax.set_title('Per-Vertebra Localization Error')
    
    # Add value labels
    for bar, med in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{med:.1f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
```

---

## Paper Writing Checklist

### Results Section

- [ ] Table 1: Main results (MAHT-Net vs Klinwichit baseline)
- [ ] Table 2: Ablation study results
- [ ] Table 3: Per-vertebra analysis
- [ ] Figure 3: Architecture diagram
- [ ] Figure 4: Training curves
- [ ] Figure 5: Ablation bar chart
- [ ] Figure 6: Qualitative results
- [ ] Figure 7: Attention visualization
- [ ] Figure 8: Failure cases

### Statistical Analysis

- [ ] Paired t-test for significance
- [ ] 95% CI for MED
- [ ] Cross-validation results (if applicable)

---

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Dataset preparation | Working dataloader |
| 2 | MAHT-Net implementation | Model training |
| 3 | Training AP view | Best model checkpoint |
| 4 | Training LA view | Both views complete |
| 5 | Ablation studies (VAM focus) | Ablation results |
| 6 | Analysis + visualization | Figures ready |
| 7 | Paper writing | Draft complete |
| 8 | Revision + submission | Final paper |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM on Colab | Reduce batch size to 4 |
| Low MED not improving | Increase VAM layers, add more augmentation |
| Training instability | Lower LR, add warmup |
| Poor generalization | Add dropout, use stronger augmentation |

### Debug Commands

```bash
# Check GPU memory
nvidia-smi

# Monitor training
tensorboard --logdir experiments/results

# Quick test run
python train_maht_net.py --epochs 2 --batch-size 2

# Profile memory usage
python -c "
import torch
from models.maht_net import MAHTNet
m = MAHTNet().cuda()
x = torch.randn(4, 3, 512, 512).cuda()
torch.cuda.reset_peak_memory_stats()
y = m(x)
print(f'Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
"
```

---

## Final Checklist Before Submission

- [ ] All experiments completed
- [ ] Results table finalized
- [ ] Statistical significance confirmed
- [ ] Code cleaned and documented
- [ ] Models uploaded to Drive/repository
- [ ] Paper proofread
- [ ] Supplementary materials ready

---

*Last Updated: February 2025*
