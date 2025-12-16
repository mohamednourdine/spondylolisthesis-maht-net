# Training Troubleshooting Guide

**Purpose**: Quick solutions to common training problems  
**Use this when**: Training fails, gives errors, or doesn't converge

---

## 🔍 Diagnostic Checklist

Before troubleshooting, run this diagnostic:

```bash
# Check environment
conda activate phd
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check data
python scripts/verify_data.py

# Check disk space
df -h | grep -E '(Filesystem|/Users)'

# Check GPU memory (if using GPU)
nvidia-smi
```

---

## ⚠️ Common Errors & Solutions

### 1. Import Errors

#### Error: `ModuleNotFoundError: No module named 'torch'`

**Cause**: Wrong environment or packages not installed

**Solution:**
```bash
# Activate correct environment
conda activate phd

# Verify installation
pip list | grep torch

# If not found, reinstall
pip install torch==1.13.0 torchvision==0.14.0
```

---

#### Error: `ModuleNotFoundError: No module named 'src'`

**Cause**: Python can't find your project modules

**Solution:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/Users/mnourdine/phd/spondylolisthesis-maht-net"

# Or install in editable mode
pip install -e .

# Or run from project root
cd /Users/mnourdine/phd/spondylolisthesis-maht-net
python scripts/train_unet.py
```

---

### 2. Data Loading Errors

#### Error: `FileNotFoundError: data/Train/images not found`

**Cause**: Data directory doesn't exist or wrong path

**Solution:**
```bash
# Check if data exists
ls -la data/Train/images/ | head -5

# If missing, check actual location
find . -name "Train" -type d

# Update config with correct path
# Edit experiments/configs/unet_config.yaml
```

---

#### Error: `JSONDecodeError: Expecting value`

**Cause**: Corrupted or invalid JSON annotation file

**Solution:**
```bash
# Find problematic files
for f in data/Train/labels/*.json; do
    python -c "import json; json.load(open('$f'))" 2>&1 | grep -q "Error" && echo "$f"
done

# Inspect specific file
cat data/Train/labels/problematic_file.json

# If corrupted, remove or fix:
# Option 1: Remove from training
rm data/Train/labels/problematic_file.json
rm data/Train/images/problematic_file.jpg

# Option 2: Fix JSON syntax manually
```

---

#### Error: `KeyError: 'keypoints'`

**Cause**: Annotation format doesn't match expected structure

**Solution:**
```python
# Check annotation structure
import json
with open('data/Train/labels/image_001.json') as f:
    data = json.load(f)
    print(data.keys())  # Should have 'boxes', 'keypoints', 'labels'
    print(f"Keypoints shape: {len(data['keypoints'])}")  # Should be 4 (vertebrae)
    print(f"Points per vertebra: {len(data['keypoints'][0])}")  # Should be 4 (corners)

# Update dataset class if structure is different
```

---

### 3. Memory Errors

#### Error: `RuntimeError: CUDA out of memory`

**Cause**: Batch size too large for GPU memory

**Solution:**
```yaml
# In experiments/configs/unet_config.yaml
training:
  batch_size: 2  # Reduce from 8 to 2
  
# Or use gradient accumulation:
training:
  batch_size: 2
  accumulation_steps: 4  # Effective batch size = 2 * 4 = 8
```

**Alternative - Use CPU:**
```python
# In training script, change:
device = torch.device('cpu')  # Instead of 'cuda'
```

---

#### Error: `RuntimeError: [enforce fail at alloc_cpu.cpp]`

**Cause**: Out of RAM (system memory)

**Solution:**
```yaml
# Reduce batch size
training:
  batch_size: 1

# Reduce number of workers
# In DataLoader:
num_workers: 0  # Instead of 4
```

---

### 4. Training Errors

#### Error: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and data on different devices (CPU vs GPU)

**Solution:**
```python
# Make sure everything is on same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
images = images.to(device)
targets = targets.to(device)

# Check in code:
print(f"Model device: {next(model.parameters()).device}")
print(f"Images device: {images.device}")
```

---

#### Error: `Loss is NaN`

**Cause**: Learning rate too high, gradient explosion, or bad data

**Solution 1 - Reduce Learning Rate:**
```yaml
# In config
training:
  learning_rate: 0.0001  # Instead of 0.001
```

**Solution 2 - Add Gradient Clipping:**
```python
# In training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Solution 3 - Check Data:**
```python
# Verify no NaN in inputs
print(f"Images has NaN: {torch.isnan(images).any()}")
print(f"Targets has NaN: {torch.isnan(targets).any()}")

# Check normalization
print(f"Image range: [{images.min()}, {images.max()}]")
```

---

#### Error: `Loss not decreasing`

**Cause**: Learning rate too low, wrong loss function, or data issues

**Solutions:**

**1. Try different learning rates:**
```yaml
learning_rate: 0.01   # Try higher
learning_rate: 0.001  # Medium
learning_rate: 0.0001 # Lower
```

**2. Check loss function:**
```python
# For keypoint detection, use:
criterion = nn.MSELoss()  # For coordinate regression

# NOT CrossEntropyLoss (for classification)
```

**3. Verify gradients are flowing:**
```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
    else:
        print(f"{name}: NO GRADIENT!")
```

**4. Check if model is in training mode:**
```python
model.train()  # Not model.eval()
```

---

### 5. Validation Errors

#### Error: `Validation metrics worse than random`

**Cause**: Model not learning or predictions incorrect

**Solutions:**

**1. Visualize predictions:**
```python
# After validation
import matplotlib.pyplot as plt

# Get one batch
images, targets = next(iter(val_loader))
images = images.to(device)

model.eval()
with torch.no_grad():
    predictions = model(images)

# Plot first image
img = images[0].cpu().permute(1, 2, 0).numpy()
pred_kpts = predictions[0].cpu().numpy()
true_kpts = targets['keypoints'][0].numpy()

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], c='red', label='Predicted')
plt.title('Predictions')
plt.subplot(122)
plt.imshow(img)
plt.scatter(true_kpts[:, :, 0], true_kpts[:, :, 1], c='green', label='Ground Truth')
plt.title('Ground Truth')
plt.show()
```

**2. Check coordinate ranges:**
```python
print(f"Predictions range: [{predictions.min()}, {predictions.max()}]")
print(f"Targets range: [{targets['keypoints'].min()}, {targets['keypoints'].max()}]")

# They should be in same range (e.g., 0-512 for 512x512 image)
```

---

#### Error: `MRE is very high (>20mm)`

**Cause**: Model predicting wrong coordinates or scale mismatch

**Solutions:**

**1. Check coordinate normalization:**
```python
# Make sure coordinates are in pixel space (0-512)
# NOT normalized (0-1)

# If needed, denormalize:
predictions = predictions * image_size  # image_size = 512
```

**2. Verify keypoint format:**
```python
# Check shape
print(f"Predictions shape: {predictions.shape}")  # Should be [batch, 16, 2]
print(f"Targets shape: {targets['keypoints'].shape}")  # Should match

# Check a sample
print(f"First prediction: {predictions[0, 0]}")  # Should be [x, y] in pixel coords
```

---

### 6. Performance Issues

#### Problem: Training is very slow

**Causes & Solutions:**

**1. Using CPU instead of GPU:**
```python
# Check device
print(torch.cuda.is_available())  # Should be True

# If False, either:
# - Your machine doesn't have GPU
# - CUDA not installed properly
# - Using CPU version of PyTorch

# Install GPU version (if you have GPU):
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

**2. Too many DataLoader workers:**
```python
# In DataLoader
num_workers=0  # For debugging
num_workers=2  # For training on laptop
num_workers=4  # For training on workstation
```

**3. No data caching:**
```python
# Cache preprocessed data
class CachedDataset:
    def __init__(self, dataset):
        self.cache = [dataset[i] for i in range(len(dataset))]
    
    def __getitem__(self, idx):
        return self.cache[idx]
    
    def __len__(self):
        return len(self.cache)
```

---

#### Problem: Training works but takes weeks

**This is normal for:**
- CPU training (1-2 weeks per model)
- Large models (Keypoint R-CNN, MAHT-Net)
- High-resolution images (1024x1024)

**Speed up options:**

**1. Use smaller images:**
```yaml
data:
  image_size: [256, 256]  # Instead of 512
```

**2. Train on fewer epochs first:**
```yaml
training:
  epochs: 20  # Quick test, then increase to 100
```

**3. Use cloud GPU:**
- Google Colab (free GPU)
- AWS/GCP/Azure (paid)
- University cluster (if available)

---

### 7. Checkpoint & Logging Errors

#### Error: `FileNotFoundError: experiments/checkpoints/unet`

**Cause**: Directory doesn't exist

**Solution:**
```bash
mkdir -p experiments/checkpoints/unet
mkdir -p experiments/logs/unet
mkdir -p experiments/results/unet
```

---

#### Error: `Permission denied` when saving checkpoint

**Cause**: No write permission

**Solution:**
```bash
# Check permissions
ls -la experiments/checkpoints/

# Fix permissions
chmod -R u+w experiments/

# Or save to different location
# Update config:
paths:
  checkpoint_dir: ~/checkpoints/unet
```

---

## 🧪 Debug Mode Training

When troubleshooting, use this minimal training loop:

```python
#!/usr/bin/env python3
"""Minimal training script for debugging."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Your imports
from models.unet import UNet
from src.data.dataset import SpondylolisthesisDataset

def debug_train():
    """Minimal training loop for debugging."""
    
    print("="*60)
    print("DEBUG MODE TRAINING")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = UNet(num_keypoints=16).to(device)
    print(f"✅ Model created")
    
    # Create dataset (just 10 samples)
    dataset = SpondylolisthesisDataset(
        image_dir='data/Train/images',
        label_dir='data/Train/labels',
        image_size=(512, 512)
    )
    subset = torch.utils.data.Subset(dataset, range(10))
    loader = DataLoader(subset, batch_size=2, shuffle=True)
    print(f"✅ Dataset created: {len(subset)} samples")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"✅ Loss and optimizer ready")
    
    # Training loop (just 3 iterations)
    print("\nStarting training...")
    model.train()
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        for batch_idx, (images, targets) in enumerate(loader):
            print(f"  Batch {batch_idx + 1}/{len(loader)}")
            
            # Move to device
            images = images.to(device)
            keypoints = targets['keypoints'].to(device)
            print(f"    Images: {images.shape}")
            print(f"    Keypoints: {keypoints.shape}")
            
            # Forward
            optimizer.zero_grad()
            predictions = model(images)
            print(f"    Predictions: {predictions.shape}")
            
            # Loss
            loss = criterion(predictions, keypoints.view(keypoints.size(0), -1, 2))
            print(f"    Loss: {loss.item():.4f}")
            
            # Backward
            loss.backward()
            optimizer.step()
            
            print(f"    ✅ Batch complete")
    
    print("\n" + "="*60)
    print("✅ DEBUG TRAINING PASSED!")
    print("="*60)
    print("\nYour setup is working correctly!")
    print("Now you can run full training.")

if __name__ == "__main__":
    debug_train()
```

**Run debug script:**
```bash
python scripts/debug_train.py
```

---

## 📋 Pre-Training Verification Script

Run this comprehensive check before starting training:

```bash
cat > scripts/pre_training_check.py << 'EOF'
#!/usr/bin/env python3
"""Comprehensive pre-training verification."""

import sys
import torch
from pathlib import Path

def check_environment():
    """Check Python environment."""
    print("\n1️⃣  ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✅ Environment OK")

def check_directories():
    """Check required directories exist."""
    print("\n2️⃣  DIRECTORY CHECK")
    print("="*60)
    
    required = [
        "data/Train/images",
        "data/Train/labels",
        "data/Validation/images",
        "data/Validation/labels",
        "experiments/checkpoints",
        "experiments/logs",
        "experiments/configs"
    ]
    
    all_good = True
    for dir_path in required:
        exists = Path(dir_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}")
        if not exists:
            all_good = False
    
    if all_good:
        print("✅ All directories OK")
    else:
        print("⚠️  Some directories missing - run setup script")
    
    return all_good

def check_data():
    """Check data files."""
    print("\n3️⃣  DATA CHECK")
    print("="*60)
    
    train_imgs = len(list(Path("data/Train/images").glob("*.jpg")))
    train_lbls = len(list(Path("data/Train/labels").glob("*.json")))
    val_imgs = len(list(Path("data/Validation/images").glob("*.jpg")))
    val_lbls = len(list(Path("data/Validation/labels").glob("*.json")))
    
    print(f"Training images: {train_imgs}")
    print(f"Training labels: {train_lbls}")
    print(f"Validation images: {val_imgs}")
    print(f"Validation labels: {val_lbls}")
    
    if train_imgs == train_lbls and val_imgs == val_lbls:
        print("✅ Data counts match")
        return True
    else:
        print("❌ Data count mismatch!")
        return False

def check_config():
    """Check config file exists."""
    print("\n4️⃣  CONFIG CHECK")
    print("="*60)
    
    config_path = Path("experiments/configs/unet_config.yaml")
    if config_path.exists():
        print(f"✅ Config found: {config_path}")
        return True
    else:
        print(f"❌ Config missing: {config_path}")
        return False

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("PRE-TRAINING VERIFICATION")
    print("="*60)
    
    results = []
    results.append(("Environment", True))  # Always passes if script runs
    results.append(("Directories", check_directories()))
    results.append(("Data", check_data()))
    results.append(("Config", check_config()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! You're ready to train!")
    else:
        print("\n⚠️  Some checks failed. Fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
EOF

python scripts/pre_training_check.py
```

---

## 🆘 Still Stuck?

If you've tried everything and still have issues:

1. **Check the logs carefully** - error messages usually tell you what's wrong
2. **Google the exact error message** - someone has likely encountered it before
3. **Simplify** - try the debug training script first
4. **Document** - write down what you tried and what happened
5. **Ask for help** - post on GitHub Issues, Stack Overflow, or PyTorch forums

**Include in your help request:**
- Full error message
- Python version, PyTorch version
- Operating system
- GPU/CPU
- What you've tried
- Minimal code to reproduce

---

## ✅ Success Indicators

Your training is working well if you see:

1. **Loss decreases steadily** (not staying flat or increasing)
2. **No NaN or Inf** in loss values
3. **Validation metrics improve** over epochs
4. **MRE < 5mm** by epoch 50 (for U-Net)
5. **SDR@2mm > 70%** by epoch 50
6. **Training doesn't crash** for multiple epochs

---

**Good luck debugging! 🔧**

Most issues are simple fixes - don't give up!
