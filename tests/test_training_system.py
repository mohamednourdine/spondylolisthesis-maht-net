"""Quick test of the training system setup."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing training system components...\n")

# Test 1: Model Registry
print("1. Testing Model Registry...")
from models.model_registry import ModelRegistry
models = ModelRegistry.list_models()
print(f"   Available models: {models}")
assert 'unet' in models
print("   ✓ Model Registry works\n")

# Test 2: Config
print("2. Testing Configuration...")
from config.unet_config import UNetConfig
config_dict = UNetConfig.to_dict()
print(f"   Config keys: {list(config_dict.keys())[:5]}...")
print(f"   Batch size: {UNetConfig.BATCH_SIZE}")
print("   ✓ Configuration works\n")

# Test 3: Model Creation
print("3. Testing Model Creation...")
import torch
model = ModelRegistry.create('unet', in_channels=3, num_keypoints=4)
print(f"   Model type: {type(model).__name__}")
dummy_input = torch.randn(1, 3, 256, 256)
output = model(dummy_input)
print(f"   Output shape: {output.shape}")
print("   ✓ Model creation works\n")

# Test 4: Loss
print("4. Testing Loss Function...")
from training.losses import UNetKeypointLoss
criterion = UNetKeypointLoss()
pred = torch.randn(2, 4, 64, 64, requires_grad=True)
target = torch.rand(2, 4, 64, 64)
loss = criterion(pred, target)
print(f"   Loss value: {loss.item():.4f}")
loss.backward()
print("   ✓ Loss function works\n")

print("="*50)
print("✓ All components working correctly!")
print("="*50)
