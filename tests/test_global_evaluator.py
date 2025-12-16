"""Test the global evaluator."""

import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing Global Keypoint Evaluator...\n")

# Test 1: Evaluator Creation
print("1. Testing Evaluator Creation...")
from evaluation.keypoint_evaluator import get_global_evaluator
evaluator = get_global_evaluator()
print(f"   SDR thresholds: {evaluator.sdr_thresholds}")
print("   ✓ Evaluator created\n")

# Test 2: Keypoint Extraction
print("2. Testing Keypoint Extraction from Heatmaps...")
batch_size = 2
num_keypoints = 4
heatmaps = torch.randn(batch_size, num_keypoints, 64, 64)
keypoints = evaluator.extract_keypoints_from_heatmaps(heatmaps)
print(f"   Extracted {len(keypoints)} images")
print(f"   First image keypoints shape: {keypoints[0].shape}")
print("   ✓ Keypoint extraction works\n")

# Test 3: Metrics Computation
print("3. Testing Metrics Computation...")
pred_kps = [np.random.rand(4, 2) * 64 for _ in range(2)]
target_kps = [torch.rand(2, 4, 3) * 64 for _ in range(2)]  # 2 vertebrae, 4 corners
metrics = evaluator.compute_metrics(pred_kps, target_kps)
print(f"   Computed metrics: {list(metrics.keys())}")
print(f"   MRE: {metrics['MRE']:.2f}")
print(f"   SDR_2.0mm: {metrics['SDR_2.0mm']:.4f}")
print("   ✓ Metrics computation works\n")

# Test 4: Batch Evaluation
print("4. Testing Batch Evaluation...")
pred_heatmaps = torch.randn(2, 4, 64, 64)
target_heatmaps = torch.rand(2, 4, 64, 64)
target_keypoints = [torch.rand(2, 4, 3) * 64 for _ in range(2)]
batch_metrics = evaluator.evaluate_batch(
    pred_heatmaps,
    target_heatmaps,
    target_keypoints
)
print(f"   Batch metrics: {batch_metrics}")
print("   ✓ Batch evaluation works\n")

# Test 5: Metrics Aggregation
print("5. Testing Metrics Aggregation...")
batch_metrics_list = [batch_metrics, batch_metrics, batch_metrics]
aggregated = evaluator.aggregate_metrics(batch_metrics_list)
print(f"   Aggregated metrics: {aggregated}")
print("   ✓ Aggregation works\n")

# Test 6: Format Metrics
print("6. Testing Metrics Formatting...")
formatted = evaluator.format_metrics(metrics, prefix='train_')
print(f"   Formatted: {formatted}")
print("   ✓ Formatting works\n")

print("="*60)
print("✓ All Global Evaluator tests passed!")
print("="*60)
