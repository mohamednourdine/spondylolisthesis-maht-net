#!/usr/bin/env python3
"""Generate figures for documentation."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent

def main():
    # Load UNet history
    unet_path = PROJECT_ROOT / 'experiments/results/unet/unet_20260113_125631/training_history.json'
    unet_h = json.load(open(unet_path))
    unet_mre = [m['MRE_px'] for m in unet_h['val_metrics']]
    unet_sdr24 = [m['SDR_24px'] * 100 for m in unet_h['val_metrics']]
    
    # Load ResNet history
    resnet_path = PROJECT_ROOT / 'experiments/results/resnet/resnet50_pretrained_20260125_223141/training_history.json'
    resnet_h = json.load(open(resnet_path))
    resnet_mre = [m['MRE_px'] for m in resnet_h['val_metrics']]
    resnet_sdr24 = [m['SDR_24px'] * 100 for m in resnet_h['val_metrics']]
    
    # Output directory
    out_dir = PROJECT_ROOT / 'docs/figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # Figure 1: Training Curves Comparison
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MRE comparison
    ax1 = axes[0]
    ax1.plot(range(1, len(unet_mre)+1), unet_mre, 'b-', label='UNet', linewidth=2)
    ax1.plot(range(1, len(resnet_mre)+1), resnet_mre, 'r-', label='ResNet-50', linewidth=2)
    ax1.axhline(y=65.07, color='b', linestyle='--', alpha=0.5, label='UNet Best (65.07 px)')
    ax1.axhline(y=51.06, color='r', linestyle='--', alpha=0.5, label='ResNet Best (51.06 px)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation MRE (px)', fontsize=12)
    ax1.set_title('Mean Radial Error Comparison', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([40, 160])
    
    # SDR@24px comparison
    ax2 = axes[1]
    ax2.plot(range(1, len(unet_sdr24)+1), unet_sdr24, 'b-', label='UNet', linewidth=2)
    ax2.plot(range(1, len(resnet_sdr24)+1), resnet_sdr24, 'r-', label='ResNet-50', linewidth=2)
    ax2.axhline(y=29.7, color='b', linestyle='--', alpha=0.5, label='UNet Best (29.7%)')
    ax2.axhline(y=36.1, color='r', linestyle='--', alpha=0.5, label='ResNet Best (36.1%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation SDR@24px (%)', fontsize=12)
    ax2.set_title('Successful Detection Rate Comparison', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 50])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves_comparison.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir}/training_curves_comparison.png')
    plt.close()
    
    # ============================================
    # Figure 2: Training Details
    # ============================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    # UNet Loss
    ax = axes2[0, 0]
    ax.plot(unet_h['train_losses'], 'b-', label='Train', alpha=0.7)
    ax.plot(unet_h['val_losses'], 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('UNet: Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ResNet Loss
    ax = axes2[0, 1]
    ax.plot(resnet_h['train_losses'], 'b-', label='Train', alpha=0.7)
    ax.plot(resnet_h['val_losses'], 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('ResNet-50: Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # UNet SDR breakdown
    ax = axes2[1, 0]
    unet_sdr6 = [m['SDR_6px'] * 100 for m in unet_h['val_metrics']]
    unet_sdr12 = [m['SDR_12px'] * 100 for m in unet_h['val_metrics']]
    unet_sdr18 = [m['SDR_18px'] * 100 for m in unet_h['val_metrics']]
    ax.plot(unet_sdr6, label='SDR@6px', alpha=0.8)
    ax.plot(unet_sdr12, label='SDR@12px', alpha=0.8)
    ax.plot(unet_sdr18, label='SDR@18px', alpha=0.8)
    ax.plot(unet_sdr24, label='SDR@24px', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SDR (%)')
    ax.set_title('UNet: SDR at Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ResNet SDR breakdown
    ax = axes2[1, 1]
    resnet_sdr6 = [m['SDR_6px'] * 100 for m in resnet_h['val_metrics']]
    resnet_sdr12 = [m['SDR_12px'] * 100 for m in resnet_h['val_metrics']]
    resnet_sdr18 = [m['SDR_18px'] * 100 for m in resnet_h['val_metrics']]
    ax.plot(resnet_sdr6, label='SDR@6px', alpha=0.8)
    ax.plot(resnet_sdr12, label='SDR@12px', alpha=0.8)
    ax.plot(resnet_sdr18, label='SDR@18px', alpha=0.8)
    ax.plot(resnet_sdr24, label='SDR@24px', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SDR (%)')
    ax.set_title('ResNet-50: SDR at Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_details.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir}/training_details.png')
    plt.close()
    
    # ============================================
    # Figure 3: Bar Chart Comparison
    # ============================================
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['MRE\n(px, lower=better)', 'SDR@6px\n(%)', 'SDR@12px\n(%)', 'SDR@18px\n(%)', 'SDR@24px\n(%)']
    unet_vals = [65.07, 17.8, 26.8, 29.3, 29.7]
    resnet_vals = [51.06, 18.3, 31.9, 34.9, 36.1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, unet_vals, width, label='UNet', color='steelblue')
    bars2 = ax.bar(x + width/2, resnet_vals, width, label='ResNet-50', color='coral')
    
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison: UNet vs ResNet-50', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison_bar.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir}/model_comparison_bar.png')
    plt.close()
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
