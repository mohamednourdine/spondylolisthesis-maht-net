"""Quick test of training with metrics."""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("Quick Training Test with Metrics")
print("="*60 + "\n")

# Test 1: Load config
print("1. Loading configuration...")
from config.unet_config import UNetConfig
config = UNetConfig()
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Image size: {config.IMAGE_SIZE}")
print(f"   Data path: {config.DATA_ROOT}")
print("   ✓ Config loaded\n")

# Test 2: Create dataset
print("2. Creating dataset...")
from src.data.unet_dataset import create_unet_dataloaders
train_loader, val_loader = create_unet_dataloaders(
    train_image_dir=config.TRAIN_IMAGE_DIR,
    train_label_dir=config.TRAIN_LABEL_DIR,
    val_image_dir=config.VAL_IMAGE_DIR,
    val_label_dir=config.VAL_LABEL_DIR,
    batch_size=2,  # Small batch for testing
    num_workers=0,
    heatmap_sigma=config.HEATMAP_SIGMA,
    output_stride=config.OUTPUT_STRIDE
)
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print("   ✓ Dataloaders created\n")

# Test 3: Get one batch
print("3. Getting one training batch...")
batch = next(iter(train_loader))
images = batch['images']
heatmaps = batch['heatmaps']
keypoints = batch['keypoints']
print(f"   Images shape: {images.shape}")
print(f"   Heatmaps shape: {heatmaps.shape}")
print(f"   Num samples with keypoints: {len(keypoints)}")
print("   ✓ Batch loaded\n")

# Test 4: Create model
print("4. Creating UNet model...")
from models.model_registry import ModelRegistry
model = ModelRegistry.create('unet', **config.get_model_config())
print(f"   Model created: {model.__class__.__name__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"   Device: {device}")
print("   ✓ Model created\n")

# Test 5: Forward pass
print("5. Testing forward pass...")
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
print(f"   Output shape: {outputs.shape}")
print("   ✓ Forward pass works\n")

# Test 6: Create evaluator
print("6. Creating global evaluator...")
from evaluation.keypoint_evaluator import get_global_evaluator
evaluator = get_global_evaluator()
print(f"   SDR thresholds: {evaluator.sdr_thresholds}")
print("   ✓ Evaluator created\n")

# Test 7: Evaluate batch
print("7. Testing batch evaluation...")
metrics = evaluator.evaluate_batch(outputs, heatmaps.to(device), keypoints)
print(f"   Metrics: {evaluator.format_metrics(metrics, prefix='test_')}")
print("   ✓ Evaluation works\n")

print("="*60)
print("✓ All components working! Ready for training.")
print("="*60)
print("\nTo start actual training, run:")
print("python train.py --model unet --epochs 2 --batch-size 4")
