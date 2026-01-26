#!/usr/bin/env python3
"""Generate training curves visualization for all models."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_history(path):
    """Load training history from JSON file."""
    with open(path) as f:
        return json.load(f)

def extract_metrics(history):
    """Extract metrics from training history."""
    val_metrics = history.get('val_metrics', [])
    train_metrics = history.get('train_metrics', [])
    
    epochs = list(range(1, len(val_metrics) + 1))
    
    # Extract MRE
    val_mre = [m.get('MRE_px', m.get('mean_radial_error', None)) for m in val_metrics]
    train_mre = [m.get('MRE_px', m.get('mean_radial_error', None)) for m in train_metrics] if train_metrics else None
    
    # Extract SDR@24px
    val_sdr = [m.get('SDR_24px', m.get('sdr_24px', None)) for m in val_metrics]
    train_sdr = [m.get('SDR_24px', m.get('sdr_24px', None)) for m in train_metrics] if train_metrics else None
    
    # Extract loss
    val_loss = history.get('val_losses', [])
    train_loss = history.get('train_losses', [])
    
    return {
        'epochs': epochs,
        'val_mre': val_mre,
        'train_mre': train_mre,
        'val_sdr': val_sdr,
        'train_sdr': train_sdr,
        'val_loss': val_loss,
        'train_loss': train_loss
    }

def main():
    # Project paths
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Find training histories
    hrnet_path = project_root / "experiments/results/hrnet/hrnet_w32_pretrained_ss_20260126_123737/training_history.json"
    resnet_dirs = list((project_root / "experiments/results/resnet").glob("*/training_history.json"))
    unet_dirs = list((project_root / "experiments/results/unet").glob("*/training_history.json"))
    
    print("Loading training histories...")
    
    models = {}
    colors = {'HRNet-W32': '#2ecc71', 'ResNet-50': '#3498db', 'UNet': '#e74c3c'}
    markers = {'HRNet-W32': 'o', 'ResNet-50': 's', 'UNet': '^'}
    
    # Load HRNet
    if hrnet_path.exists():
        models['HRNet-W32'] = extract_metrics(load_history(hrnet_path))
        print(f"  HRNet-W32: {len(models['HRNet-W32']['epochs'])} epochs")
    
    # Load ResNet (use latest)
    if resnet_dirs:
        models['ResNet-50'] = extract_metrics(load_history(sorted(resnet_dirs)[-1]))
        print(f"  ResNet-50: {len(models['ResNet-50']['epochs'])} epochs")
    
    # Load UNet (use latest)
    if unet_dirs:
        models['UNet'] = extract_metrics(load_history(sorted(unet_dirs)[-1]))
        print(f"  UNet: {len(models['UNet']['epochs'])} epochs")
    
    # ====================
    # Figure 1: Combined Model Comparison (2x2 grid)
    # ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves Comparison: All Models', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation MRE
    ax = axes[0, 0]
    for name, data in models.items():
        if data['val_mre'] and any(v is not None for v in data['val_mre']):
            mre = [v if v and v < 300 else np.nan for v in data['val_mre']]
            ax.plot(data['epochs'], mre, label=name, color=colors[name], linewidth=2, marker=markers[name], markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MRE (pixels)')
    ax.set_title('Validation MRE (↓ lower is better)')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation SDR@24px
    ax = axes[0, 1]
    for name, data in models.items():
        if data['val_sdr'] and any(v is not None for v in data['val_sdr']):
            sdr = [v * 100 if v else np.nan for v in data['val_sdr']]
            ax.plot(data['epochs'], sdr, label=name, color=colors[name], linewidth=2, marker=markers[name], markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SDR@24px (%)')
    ax.set_title('Validation SDR@24px (↑ higher is better)')
    ax.legend()
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Loss
    ax = axes[1, 0]
    for name, data in models.items():
        if data['val_loss']:
            ax.plot(data['epochs'], data['val_loss'], label=name, color=colors[name], linewidth=2, marker=markers[name], markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss (↓ lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Loss
    ax = axes[1, 1]
    for name, data in models.items():
        if data['train_loss']:
            ax.plot(data['epochs'], data['train_loss'], label=name, color=colors[name], linewidth=2, marker=markers[name], markevery=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_curves_all_models.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {figures_dir / 'training_curves_all_models.png'}")
    
    # ====================
    # Figure 2: Individual HRNet Curves
    # ====================
    if 'HRNet-W32' in models:
        data = models['HRNet-W32']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('HRNet-W32 Training Curves', fontsize=14, fontweight='bold')
        
        # MRE
        ax = axes[0, 0]
        mre = [v if v and v < 300 else np.nan for v in data['val_mre']]
        ax.plot(data['epochs'], mre, 'g-', linewidth=2, label='Validation')
        if data['train_mre']:
            train_mre = [v if v and v < 300 else np.nan for v in data['train_mre']]
            ax.plot(data['epochs'], train_mre, 'g--', alpha=0.5, label='Training')
        ax.axhline(y=43.85, color='r', linestyle=':', label='Best: 43.85 px')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MRE (pixels)')
        ax.set_title('Mean Radial Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # SDR
        ax = axes[0, 1]
        sdr = [v * 100 if v else np.nan for v in data['val_sdr']]
        ax.plot(data['epochs'], sdr, 'g-', linewidth=2, label='Validation')
        if data['train_sdr']:
            train_sdr = [v * 100 if v else np.nan for v in data['train_sdr']]
            ax.plot(data['epochs'], train_sdr, 'g--', alpha=0.5, label='Training')
        ax.axhline(y=43.65, color='r', linestyle=':', label='Best: 43.65%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SDR@24px (%)')
        ax.set_title('Success Detection Rate @ 24px')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[1, 0]
        ax.plot(data['epochs'], data['val_loss'], 'g-', linewidth=2, label='Validation')
        ax.plot(data['epochs'], data['train_loss'], 'g--', alpha=0.5, label='Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Train vs Val Loss gap
        ax = axes[1, 1]
        gap = [t - v for t, v in zip(data['train_loss'], data['val_loss'])]
        ax.plot(data['epochs'], gap, 'purple', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.fill_between(data['epochs'], gap, 0, alpha=0.3, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss - Val Loss')
        ax.set_title('Generalization Gap (negative = overfitting)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'hrnet_training_curves.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {figures_dir / 'hrnet_training_curves.png'}")
    
    # ====================
    # Figure 3: Final Bar Comparison
    # ====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Final Model Performance Comparison', fontsize=14, fontweight='bold')
    
    model_names = list(models.keys())
    
    # Best MRE
    ax = axes[0]
    best_mre = []
    for name in model_names:
        mre_vals = [v for v in models[name]['val_mre'] if v and v < 300]
        best_mre.append(min(mre_vals) if mre_vals else 0)
    bars = ax.bar(model_names, best_mre, color=[colors[n] for n in model_names], edgecolor='black')
    ax.set_ylabel('MRE (pixels)')
    ax.set_title('Best Validation MRE (↓ lower is better)')
    for bar, val in zip(bars, best_mre):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', fontweight='bold')
    ax.set_ylim(0, max(best_mre) * 1.15)
    
    # Best SDR@24px
    ax = axes[1]
    best_sdr = []
    for name in model_names:
        sdr_vals = [v * 100 for v in models[name]['val_sdr'] if v]
        best_sdr.append(max(sdr_vals) if sdr_vals else 0)
    bars = ax.bar(model_names, best_sdr, color=[colors[n] for n in model_names], edgecolor='black')
    ax.set_ylabel('SDR@24px (%)')
    ax.set_title('Best Validation SDR@24px (↑ higher is better)')
    for bar, val in zip(bars, best_sdr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_ylim(0, max(best_sdr) * 1.15)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_comparison_bar.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'model_comparison_bar.png'}")
    
    print("\n✓ All training curves generated!")

if __name__ == '__main__':
    main()
