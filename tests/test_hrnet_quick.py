"""Quick test of HRNet-W32 training setup."""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("Quick HRNet-W32 Training Test")
print("="*60 + "\n")

# Test 1: Load config
print("1. Loading configuration...")
from config.mac_config import MacConfig as config
print(f"   Batch size: 4 (HRNet default)")
print(f"   Image size: 512")
print(f"   Num keypoints: {config.NUM_KEYPOINTS}")
print("   ✓ Config loaded\n")

# Test 2: Create HRNet model
print("2. Creating HRNet-W32 model...")
from models.hrnet_heatmap import create_hrnet_heatmap
model = create_hrnet_heatmap(
    num_keypoints=config.NUM_KEYPOINTS,
    pretrained=True,
    freeze_backbone_stages=0,
    dropout_rate=0.3,
    output_size=512
)
device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"   Device: {device}")
print("   ✓ Model created\n")

# Test 3: Forward pass
print("3. Testing forward pass...")
x = torch.randn(2, 3, 512, 512).to(device)
with torch.no_grad():
    outputs = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {outputs.shape}")
print("   ✓ Forward pass works\n")

# Test 4: Model registry
print("4. Testing model registry...")
from models.model_registry import ModelRegistry
print(f"   Available models: {ModelRegistry.list_models()}")
model2 = ModelRegistry.create('hrnet', num_keypoints=40, pretrained=False)
print("   ✓ Registry works\n")

# Test 5: Create dataset
print("5. Creating dataset...")
from src.data.unet_dataset import create_unet_dataloaders
train_loader, val_loader = create_unet_dataloaders(
    train_image_dir=config.TRAIN_IMAGE_DIR,
    train_label_dir=config.TRAIN_LABEL_DIR,
    val_image_dir=config.VAL_IMAGE_DIR,
    val_label_dir=config.VAL_LABEL_DIR,
    batch_size=2,
    num_workers=0,
    heatmap_sigma=config.HEATMAP_SIGMA,
    output_stride=config.OUTPUT_STRIDE
)
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print("   ✓ Dataloaders created\n")

# Test 6: Get one batch and forward
print("6. Testing with real data...")
batch = next(iter(train_loader))
images = batch['images'].to(device)
heatmaps = batch['heatmaps'].to(device)
with torch.no_grad():
    outputs = model(images)
print(f"   Batch images: {images.shape}")
print(f"   Batch heatmaps: {heatmaps.shape}")
print(f"   Model output: {outputs.shape}")
print("   ✓ Real data forward pass works\n")

# Test 7: Loss computation
print("7. Testing loss computation...")
from training.losses import MSEWithWeightedBackground
criterion = MSEWithWeightedBackground()
loss = criterion(outputs, heatmaps)
print(f"   Loss value: {loss.item():.6f}")
print("   ✓ Loss computation works\n")

# Test 8: Backward pass
print("8. Testing backward pass...")
model.train()
outputs = model(images)
loss = criterion(outputs, heatmaps)
loss.backward()
print(f"   Backward loss: {loss.item():.6f}")
print("   ✓ Gradients computed\n")

# Test 9: Optimizer with differential LR
print("9. Testing optimizer setup...")
param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
optimizer.step()
optimizer.zero_grad()
for pg in param_groups:
    n = sum(p.numel() for p in pg['params'])
    print(f"   {pg['name']}: {n/1e6:.2f}M params, lr={pg['lr']}")
print("   ✓ Optimizer works\n")

# Test 10: Evaluator
print("10. Testing evaluator...")
from evaluation.keypoint_evaluator import get_global_evaluator
evaluator = get_global_evaluator(sdr_thresholds_px=config.SDR_THRESHOLDS_PX)
model.eval()
with torch.no_grad():
    outputs = model(images)
metrics = evaluator.evaluate_batch(outputs, heatmaps, batch['keypoints'])
mre = metrics.get('mean_radial_error', None)
det_rate = metrics.get('detection_rate', None)
if mre is not None:
    print(f"   MRE: {mre:.2f} px")
else:
    print("   MRE: N/A")
if det_rate is not None:
    print(f"   Detection rate: {det_rate*100:.1f}%")
else:
    print("   Detection rate: N/A")
print("   ✓ Evaluator works\n")

print("="*60)
print("All tests passed! HRNet-W32 is ready for training.")
print("="*60)
print("\nTo start full training:")
print("  python train_hrnet.py --epochs 100 --batch-size 4")
print("\nOr with multi-scale fusion:")
print("  python train_hrnet.py --epochs 100 --batch-size 4 --multi-scale")
