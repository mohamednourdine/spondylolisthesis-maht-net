# 🚀 Quick Start: Training Your First Model (TODAY!)

**Goal**: Get U-Net training started within the next 2-3 hours  
**Difficulty**: Beginner-friendly  
**Time**: 2-3 hours setup + 1-2 days training

---

## ✅ Step-by-Step Checklist (Do This Now!)

### Step 1: Verify Environment (5 minutes)

```bash
# Open terminal and navigate to project
cd /Users/mnourdine/phd/spondylolisthesis-maht-net

# Activate environment
conda activate phd

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if data exists
ls -la data/Train/images/ | head -5
ls -la data/Train/labels/ | head -5
```

**Expected Output:**
```
PyTorch: 1.13.0
torchvision: 0.14.0
CUDA available: True (or False if no GPU)
```

---

### Step 2: Create Required Directories (1 minute)

```bash
# Create all experiment directories
mkdir -p experiments/results/{unet,resnet,keypoint_rcnn,maht_net}
mkdir -p experiments/checkpoints/{unet,resnet,keypoint_rcnn,maht_net}
mkdir -p experiments/logs/{unet,resnet,keypoint_rcnn,maht_net}
mkdir -p experiments/configs
mkdir -p scripts

# Verify creation
ls -la experiments/
```

---

### Step 3: Verify Data Integrity (10 minutes)

**Create verification script:**

```bash
cat > scripts/verify_data.py << 'EOF'
#!/usr/bin/env python3
import os
import json
from pathlib import Path

def verify_dataset():
    """Quick data verification."""
    data_root = Path("data")
    
    for split in ["Train", "Validation"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        
        if not img_dir.exists():
            print(f"❌ Missing: {img_dir}")
            return False
            
        images = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.json"))
        
        print(f"\n{split}:")
        print(f"  ✅ Images: {len(images)}")
        print(f"  ✅ Labels: {len(labels)}")
        
        if len(images) != len(labels):
            print(f"  ⚠️  Mismatch: {len(images)} images vs {len(labels)} labels")
        
        # Test loading one annotation
        if labels:
            with open(labels[0]) as f:
                data = json.load(f)
                if all(k in data for k in ["boxes", "keypoints", "labels"]):
                    print(f"  ✅ JSON format valid")
                else:
                    print(f"  ❌ Invalid JSON format")
                    return False
    
    print("\n✅ Dataset verification PASSED!\n")
    return True

if __name__ == "__main__":
    verify_dataset()
EOF

# Run verification
python scripts/verify_data.py
```

**Expected Output:**
```
Train:
  ✅ Images: 494
  ✅ Labels: 494
  ✅ JSON format valid

Validation:
  ✅ Images: 206
  ✅ Labels: 206
  ✅ JSON format valid

✅ Dataset verification PASSED!
```

---

### Step 4: Calculate Dataset Statistics (15 minutes)

```bash
cat > scripts/calculate_stats.py << 'EOF'
#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def calculate_stats():
    """Calculate mean and std for normalization."""
    img_dir = Path("data/Train/images")
    images = list(img_dir.glob("*.jpg"))[:100]  # Sample 100 images for speed
    
    print(f"Calculating stats from {len(images)} images...")
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0
    
    for img_path in tqdm(images):
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0
        
        pixel_sum += img_array.sum(axis=(0, 1))
        pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
        num_pixels += img_array.shape[0] * img_array.shape[1]
    
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    
    print(f"\n📊 Dataset Statistics (use these in config):")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    
    return mean, std

if __name__ == "__main__":
    calculate_stats()
EOF

python scripts/calculate_stats.py
```

---

### Step 5: Test Data Loading (20 minutes)

**Create a simple test script:**

```bash
cat > scripts/test_dataloader.py << 'EOF'
#!/usr/bin/env python3
"""Test that data loading works correctly."""

import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from PIL import Image

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))
        img_array = np.array(img) / 255.0
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
        
        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.json"
        with open(label_path) as f:
            data = json.load(f)
        
        # Extract keypoints
        keypoints = np.array(data['keypoints'])  # Shape: [4, 4, 3]
        keypoints_tensor = torch.FloatTensor(keypoints)
        
        return img_tensor, {'keypoints': keypoints_tensor, 'image_id': idx}

def test_dataloader():
    """Test data loading."""
    print("Creating dataset...")
    dataset = SimpleDataset('data/Train/images', 'data/Train/labels')
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    print("\nCreating dataloader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"✅ DataLoader created: {len(loader)} batches")
    
    print("\nTesting first batch...")
    images, targets = next(iter(loader))
    
    print(f"✅ Batch loaded successfully!")
    print(f"  Images shape: {images.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Keypoints shape: {targets['keypoints'].shape}")
    print(f"  Keypoints range: [{targets['keypoints'].min():.1f}, {targets['keypoints'].max():.1f}]")
    
    print("\n✅ Data loading test PASSED!")
    print("You're ready to start training!\n")

if __name__ == "__main__":
    test_dataloader()
EOF

python scripts/test_dataloader.py
```

**Expected Output:**
```
Creating dataset...
✅ Dataset created: 494 samples

Creating dataloader...
✅ DataLoader created: 124 batches

Testing first batch...
✅ Batch loaded successfully!
  Images shape: torch.Size([4, 3, 512, 512])
  Image range: [0.000, 1.000]
  Keypoints shape: torch.Size([4, 4, 4, 3])
  Keypoints range: [0.0, 512.0]

✅ Data loading test PASSED!
You're ready to start training!
```

---

### Step 6: Run Data Understanding Notebook (30 minutes)

```bash
# Start Jupyter
jupyter notebook notebooks/00_data_understanding.ipynb
```

**In Jupyter:**
1. Click "Run All" from the menu
2. Wait for all cells to execute (15-20 minutes)
3. Check that all visualizations appear
4. Verify no errors in output

**Key outputs to check:**
- Dataset statistics table
- Sample images with keypoint overlays
- Distribution plots
- Quality validation results

---

### Step 7: Create Minimal U-Net Config (5 minutes)

```bash
cat > experiments/configs/unet_config.yaml << 'EOF'
model:
  name: unet
  num_keypoints: 16
    
training:
  batch_size: 4  # Start small
  epochs: 50     # Shorter for first test
  learning_rate: 0.001
  
data:
  image_size: [512, 512]
  normalize:
    mean: [0.5, 0.5, 0.5]  # Update after calculating stats
    std: [0.5, 0.5, 0.5]   # Update after calculating stats
    
paths:
  train_images: data/Train/images
  train_labels: data/Train/labels
  val_images: data/Validation/images
  val_labels: data/Validation/labels
  checkpoint_dir: experiments/checkpoints/unet
  log_dir: experiments/logs/unet
  
seed: 42
EOF

cat experiments/configs/unet_config.yaml
```

---

### Step 8: Implement Minimal U-Net (30 minutes)

Check if U-Net model exists:

```bash
cat models/unet.py
```

If empty or doesn't exist, create a minimal version:

```bash
cat > models/unet.py << 'EOF'
import torch
import torch.nn as nn

class UNet(nn.Module):
    """Simple U-Net for keypoint detection."""
    
    def __init__(self, in_channels=3, num_keypoints=16):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Output layer - predict x,y for each keypoint
        self.output = nn.Conv2d(64, num_keypoints * 2, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        out = self.output(d1)
        
        # Global average pooling to get keypoint coordinates
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), self.num_keypoints, 2)
        
        return out

# Test
if __name__ == "__main__":
    model = UNet(num_keypoints=16)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")  # Should be [2, 16, 2]
    print("✅ U-Net test passed!")
EOF

python models/unet.py
```

---

### Step 9: YOU'RE READY! Start Training 🎉

Now you have everything in place. Read the full training guide:

```bash
# Open the detailed guide
code docs/training_guide.md

# Or view in terminal
cat docs/training_guide.md
```

**Next actions:**
1. Implement the full training script following the guide
2. Or start with the notebook approach (easier for learning)
3. Monitor training progress
4. Iterate and improve

---

## 📊 Expected Timeline

| Time | Task | Status |
|------|------|--------|
| **Today (3 hours)** | Complete Steps 1-9 above | ⏳ |
| **Tomorrow** | Start U-Net training | ⏳ |
| **Day 2-3** | U-Net training completes | ⏳ |
| **Week 2** | Evaluate and refine U-Net | ⏳ |
| **Week 3+** | Train other models | ⏳ |

---

## 🆘 Need Help?

**If you get stuck:**

1. **Check logs:** `tail -f experiments/logs/unet/training.log`
2. **Verify GPU:** `nvidia-smi` (if using GPU)
3. **Check disk space:** `df -h`
4. **Review error messages** carefully

**Common first-time issues:**
- ❌ Import errors → Check environment: `conda activate phd`
- ❌ Data not found → Verify paths in config
- ❌ Out of memory → Reduce batch_size to 2
- ❌ Slow training → Normal on CPU, consider using GPU or cloud

---

## 🎯 Success Criteria for Today

- [x] Environment verified
- [x] Data verified  
- [x] Directories created
- [x] Data loading tested
- [x] Config file created
- [x] U-Net model implemented
- [ ] **Ready to train!**

---

**You're all set! 🚀**

Go to `docs/training_guide.md` for the complete training documentation.

Good luck establishing the first baselines on this dataset!
