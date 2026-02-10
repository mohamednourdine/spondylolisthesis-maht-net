# Step 4: Evaluation Protocol

## Overview

This guide covers the evaluation methodology for MAHT-Net, ensuring fair comparison with baseline methods and reproducible results.

---

## Evaluation Metrics

### Primary Metric: Mean Euclidean Distance (MED)

The MED measures the average pixel distance between predicted and ground-truth keypoints:

$$\text{MED} = \frac{1}{N \times K} \sum_{i=1}^{N} \sum_{k=1}^{K} \sqrt{(x_{ik}^{pred} - x_{ik}^{gt})^2 + (y_{ik}^{pred} - y_{ik}^{gt})^2}$$

Where:
- $N$ = number of images
- $K$ = number of keypoints (20 for AP, 22 for LA)
- $(x, y)^{pred}$ = predicted coordinates
- $(x, y)^{gt}$ = ground-truth coordinates

**Convert to mm** using pixel spacing:
$$\text{MED}_{mm} = \text{MED}_{pixels} \times \text{pixel\_spacing}$$

### Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **PCK@τ** | % of keypoints within τ pixels | Success rate |
| **SDR** | Standard deviation of distances | Consistency |
| **Max Error** | Max distance across all keypoints | Worst case |
| **Per-vertebra MED** | MED for each L1-L5 | Localization |

---

## Evaluation Implementation

Create `evaluation/metrics.py`:

```python
"""
Evaluation metrics for keypoint detection.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


def compute_med(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    pixel_spacing: float = 1.0
) -> float:
    """
    Compute Mean Euclidean Distance.
    
    Args:
        pred_keypoints: (N, K, 2) predicted coordinates
        gt_keypoints: (N, K, 2) ground-truth coordinates
        pixel_spacing: mm per pixel (for converting to mm)
        
    Returns:
        MED in the specified unit (pixels if pixel_spacing=1.0)
    """
    # Euclidean distance for each keypoint
    distances = torch.sqrt(
        (pred_keypoints[:, :, 0] - gt_keypoints[:, :, 0])**2 +
        (pred_keypoints[:, :, 1] - gt_keypoints[:, :, 1])**2
    )
    
    # Mean across all keypoints and images
    med = distances.mean().item() * pixel_spacing
    
    return med


def compute_pck(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    threshold: float = 10.0
) -> float:
    """
    Compute Percentage of Correct Keypoints (PCK).
    
    Args:
        pred_keypoints: (N, K, 2) predicted coordinates
        gt_keypoints: (N, K, 2) ground-truth coordinates  
        threshold: Distance threshold in pixels
        
    Returns:
        PCK percentage (0-1)
    """
    distances = torch.sqrt(
        (pred_keypoints[:, :, 0] - gt_keypoints[:, :, 0])**2 +
        (pred_keypoints[:, :, 1] - gt_keypoints[:, :, 1])**2
    )
    
    correct = (distances < threshold).float()
    pck = correct.mean().item()
    
    return pck


def compute_per_vertebra_med(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    view: str = 'AP'
) -> Dict[str, float]:
    """
    Compute MED for each vertebra separately.
    
    Args:
        pred_keypoints: (N, K, 2) predicted coordinates
        gt_keypoints: (N, K, 2) ground-truth coordinates
        view: 'AP' (K=20) or 'LA' (K=22)
        
    Returns:
        Dict mapping vertebra name to MED
    """
    K = pred_keypoints.shape[1]
    
    # Keypoint mapping (4 keypoints per vertebra for AP)
    if view == 'AP':
        vertebrae = ['L1', 'L2', 'L3', 'L4', 'L5']
        kps_per_vertebra = 4
    else:  # LA
        vertebrae = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
        kps_per_vertebra = 4  # S1 has only 2, handled separately
    
    results = {}
    
    for i, vertebra in enumerate(vertebrae):
        if view == 'LA' and vertebra == 'S1':
            start_idx = 20
            end_idx = 22
        else:
            start_idx = i * kps_per_vertebra
            end_idx = start_idx + kps_per_vertebra
        
        # Extract keypoints for this vertebra
        pred_v = pred_keypoints[:, start_idx:end_idx]
        gt_v = gt_keypoints[:, start_idx:end_idx]
        
        # Compute MED
        distances = torch.sqrt(
            (pred_v[:, :, 0] - gt_v[:, :, 0])**2 +
            (pred_v[:, :, 1] - gt_v[:, :, 1])**2
        )
        results[vertebra] = distances.mean().item()
    
    return results


def compute_sdr(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    thresholds: list = [2, 4, 6, 8, 10]
) -> Dict[float, float]:
    """
    Compute Success Detection Rate (SDR) at multiple thresholds.
    
    Args:
        pred_keypoints: (N, K, 2)
        gt_keypoints: (N, K, 2)
        thresholds: List of distance thresholds in pixels
        
    Returns:
        Dict mapping threshold to success rate
    """
    distances = torch.sqrt(
        (pred_keypoints[:, :, 0] - gt_keypoints[:, :, 0])**2 +
        (pred_keypoints[:, :, 1] - gt_keypoints[:, :, 1])**2
    )
    
    results = {}
    for t in thresholds:
        rate = (distances < t).float().mean().item() * 100
        results[t] = rate
    
    return results


class KeypointEvaluator:
    """
    Comprehensive evaluator for keypoint detection.
    """
    
    def __init__(
        self,
        pixel_spacing: float = 0.2,  # mm per pixel (default estimate)
        view: str = 'AP'
    ):
        self.pixel_spacing = pixel_spacing
        self.view = view
        
        # Accumulate predictions
        self.all_preds = []
        self.all_gts = []
        self.image_paths = []
    
    def add_batch(
        self,
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        image_paths: list = None
    ):
        """Add a batch of predictions."""
        self.all_preds.append(pred_keypoints.cpu())
        self.all_gts.append(gt_keypoints.cpu())
        if image_paths:
            self.image_paths.extend(image_paths)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        # Concatenate all batches
        pred = torch.cat(self.all_preds, dim=0)
        gt = torch.cat(self.all_gts, dim=0)
        
        results = {}
        
        # MED in pixels
        results['med_pixels'] = compute_med(pred, gt, pixel_spacing=1.0)
        
        # MED in mm
        results['med_mm'] = compute_med(pred, gt, pixel_spacing=self.pixel_spacing)
        
        # PCK at different thresholds
        for t in [5, 10, 20]:
            results[f'pck@{t}'] = compute_pck(pred, gt, threshold=t) * 100
        
        # SDR
        sdr = compute_sdr(pred, gt)
        for t, rate in sdr.items():
            results[f'sdr@{t}px'] = rate
        
        # Per-vertebra MED
        vertebra_med = compute_per_vertebra_med(pred, gt, self.view)
        for v, med in vertebra_med.items():
            results[f'med_{v}'] = med
        
        # Standard deviation
        distances = torch.sqrt(
            (pred[:, :, 0] - gt[:, :, 0])**2 +
            (pred[:, :, 1] - gt[:, :, 1])**2
        )
        results['std_pixels'] = distances.std().item()
        results['max_error_pixels'] = distances.max().item()
        
        return results
    
    def print_report(self):
        """Print formatted evaluation report."""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📏 Distance Metrics:")
        print(f"   MED: {metrics['med_pixels']:.2f} pixels ({metrics['med_mm']:.2f} mm)")
        print(f"   Std: {metrics['std_pixels']:.2f} pixels")
        print(f"   Max Error: {metrics['max_error_pixels']:.2f} pixels")
        
        print(f"\n✓ Success Rates (PCK):")
        print(f"   PCK@5px:  {metrics['pck@5']:.1f}%")
        print(f"   PCK@10px: {metrics['pck@10']:.1f}%")
        print(f"   PCK@20px: {metrics['pck@20']:.1f}%")
        
        print(f"\n📊 Per-Vertebra MED (pixels):")
        for v in ['L1', 'L2', 'L3', 'L4', 'L5']:
            print(f"   {v}: {metrics[f'med_{v}']:.2f}")
        
        print("\n" + "="*60)
        
        return metrics
```

---

## Evaluation Script

Create `scripts/evaluate_model.py`:

```python
#!/usr/bin/env python3
"""
Evaluate a trained MAHT-Net model on test set.
"""

import argparse
import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.maht_net import create_maht_net
from src.data.buu_lspine_dataset import BUULSpineDataset
from evaluation.metrics import KeypointEvaluator


def evaluate(
    model_path: str,
    data_dir: str,
    view: str = 'AP',
    batch_size: int = 8,
    pixel_spacing: float = 0.2,
    save_predictions: bool = True
):
    """Run full evaluation on test set."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    num_keypoints = 20 if view == 'AP' else 22
    model = create_maht_net(view=view)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = BUULSpineDataset(data_dir, view=view, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluator
    evaluator = KeypointEvaluator(pixel_spacing=pixel_spacing, view=view)
    
    # Run inference
    print(f"\nEvaluating on {len(test_dataset)} test images...")
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            gt_keypoints = batch['keypoints']
            
            # Predict
            outputs = model(images)
            pred_keypoints = model.extract_keypoints(outputs['heatmaps'])
            
            # Add to evaluator
            evaluator.add_batch(pred_keypoints.cpu(), gt_keypoints)
            
            # Save predictions
            if save_predictions:
                for i in range(len(images)):
                    all_predictions.append({
                        'image_path': batch['meta']['image_path'][i],
                        'pred_keypoints': pred_keypoints[i].cpu().numpy().tolist(),
                        'gt_keypoints': gt_keypoints[i].numpy().tolist()
                    })
    
    # Print report
    metrics = evaluator.print_report()
    
    # Compare with baseline
    print("\n📈 Comparison with Baseline (Klinwichit et al.):")
    baseline_med = 4.63 if view == 'AP' else 4.91
    improvement = (baseline_med - metrics['med_mm']) / baseline_med * 100
    print(f"   Baseline MED: {baseline_med:.2f} mm")
    print(f"   Our MED: {metrics['med_mm']:.2f} mm")
    print(f"   Improvement: {improvement:+.1f}%")
    
    target_met = metrics['med_mm'] < 4.0
    print(f"\n{'✅' if target_met else '❌'} Target (MED < 4.0mm): {'MET' if target_met else 'NOT MET'}")
    
    # Save results
    if save_predictions:
        output_dir = Path(model_path).parent
        
        # Save metrics
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        with open(output_dir / 'test_predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', default='data/buu-lspine')
    parser.add_argument('--view', default='AP', choices=['AP', 'LA'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pixel-spacing', type=float, default=0.2)
    args = parser.parse_args()
    
    evaluate(
        args.model, args.data_dir, args.view,
        args.batch_size, args.pixel_spacing
    )
```

---

## Visualization Tools

Create `scripts/visualize_predictions.py`:

```python
#!/usr/bin/env python3
"""Visualize model predictions."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def visualize_sample(
    image_path: str,
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    save_path: str = None
):
    """
    Visualize predictions vs ground truth on a single image.
    """
    from PIL import Image
    
    img = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(img)
    axes[1].scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], 
                    c='red', s=30, label='Predicted')
    axes[1].set_title('Predictions')
    axes[1].axis('off')
    
    # Overlay (GT in green, Pred in red)
    axes[2].imshow(img)
    axes[2].scatter(gt_keypoints[:, 0], gt_keypoints[:, 1],
                    c='green', s=30, marker='o', label='Ground Truth')
    axes[2].scatter(pred_keypoints[:, 0], pred_keypoints[:, 1],
                    c='red', s=30, marker='x', label='Predicted')
    # Draw lines between GT and Pred
    for gt, pred in zip(gt_keypoints, pred_keypoints):
        axes[2].plot([gt[0], pred[0]], [gt[1], pred[1]], 'b-', alpha=0.5)
    axes[2].legend()
    axes[2].set_title('Comparison')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_error_heatmap(
    predictions_file: str,
    output_path: str = 'error_heatmap.png'
):
    """
    Create heatmap showing error distribution across keypoints.
    """
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    # Compute errors for each keypoint
    K = len(predictions[0]['pred_keypoints'])
    errors = np.zeros((len(predictions), K))
    
    for i, pred in enumerate(predictions):
        gt = np.array(pred['gt_keypoints'])
        pr = np.array(pred['pred_keypoints'])
        errors[i] = np.sqrt(np.sum((gt - pr)**2, axis=1))
    
    # Heatmap of errors
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(errors.T, aspect='auto', cmap='hot')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Keypoint Index')
    ax.set_title('Error Distribution Across Keypoints')
    
    plt.colorbar(im, label='Error (pixels)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved error heatmap to {output_path}")
```

---

## Baseline Comparison Table

| Method | View | MED (mm) | PCK@10 | Source |
|--------|------|----------|--------|--------|
| Klinwichit ResNet152V2 | AP | 4.63 | - | Baseline |
| Klinwichit ResNet152V2 | LA | 4.91 | - | Baseline |
| **MAHT-Net (Ours)** | AP | **< 4.0** | **> 95%** | Target |
| **MAHT-Net (Ours)** | LA | **< 4.0** | **> 95%** | Target |

---

## Statistical Significance Testing

```python
from scipy import stats

def paired_t_test(errors_baseline, errors_ours):
    """
    Test if our method is significantly better than baseline.
    
    H0: No difference between methods
    H1: Our method has lower error
    """
    t_stat, p_value = stats.ttest_rel(errors_baseline, errors_ours, alternative='greater')
    
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p < 0.05): {p_value < 0.05}")
    
    return t_stat, p_value
```

---

## Evaluation Commands

```bash
# Evaluate best model on test set
python scripts/evaluate_model.py \
    --model experiments/results/maht_net_ap/best_model.pth \
    --view AP \
    --data-dir data/buu-lspine

# Visualize predictions
python scripts/visualize_predictions.py \
    --predictions experiments/results/maht_net_ap/test_predictions.json \
    --num-samples 10 \
    --output experiments/visualizations/

# Compare with baseline
python scripts/compare_with_baseline.py \
    --ours experiments/results/maht_net_ap/test_metrics.json \
    --baseline-med 4.63
```

---

## Next Step

After evaluation, proceed to:
- [05_experiments_checklist.md](05_experiments_checklist.md) - Ablation studies and experiments

---

*Last Updated: February 2025*
