# HRNet-W32 Implementation Plan

## 📋 Overview

This document outlines the implementation plan for adding **HRNet-W32** as the third model for vertebral landmark detection. HRNet (High-Resolution Network) maintains high-resolution representations throughout the network, making it ideal for dense prediction tasks like keypoint detection.

**Reference Paper**: "Deep High-Resolution Representation Learning for Human Pose Estimation" (CVPR 2019)

---

## 1. Current State Analysis

### 1.1 Model Comparison (Updated Results)

| Model | Params | Best MRE | SDR@24px | Pretrained | Training Time |
|-------|--------|----------|----------|------------|---------------|
| UNet (baseline) | 8.65M | 48.41 px | 38.0% | No | ~4 hrs |
| ResNet-50 | 30.79M | 51.06 px | 36.1% | Yes (ImageNet) | ~6 hrs |
| **HRNet-W32** | **31.78M** | **43.85 px** ✅ | **43.65%** ✅ | Yes (ImageNet) | **4.71 hrs** |

### 1.2 Why HRNet-W32?

| Advantage | Description |
|-----------|-------------|
| **Multi-resolution parallel streams** | Maintains high-resolution features throughout, unlike ResNet which downsamples then upsamples |
| **Repeated multi-scale fusion** | Exchanges information between resolutions at every stage |
| **Proven for keypoints** | State-of-the-art on COCO keypoint detection (AP 74.4% at 256×192) |
| **Right size** | 28.5M params - between UNet (17M) and ResNet-50 (38M) |
| **ImageNet pretrained** | Strong initialization for better convergence |

### 1.3 HRNet vs Other Options Considered

| Model | Params | Why/Why Not |
|-------|--------|-------------|
| HRNet-W18 | ~21M | Smaller but less accurate |
| **HRNet-W32** | **~28.5M** | **Best balance of size/accuracy** ✅ |
| HRNet-W48 | ~63.6M | Larger, may overfit on small dataset |
| ViTPose-S | ~24M | Transformer-based, more complex, future work |

---

## 2. HRNet Architecture Overview

### 2.1 Key Architectural Differences from ResNet

```
ResNet-50 (Sequential, Single Resolution):
─────────────────────────────────────────
Input → [1/4] → [1/8] → [1/16] → [1/32] → Upsample → Output
        High     ↓        ↓         ↓         ↑
        Res    Med      Low     V.Low    Back to High

HRNet (Parallel, Multi-Resolution):
─────────────────────────────────────────────────────────────────
                    Stage 1      Stage 2       Stage 3      Stage 4
Input → Stem    ┌─ [1/4]  ─────  [1/4]  ─────  [1/4]  ────  [1/4]  ──┐
                │                  ↕            ↕ ↕          ↕ ↕ ↕    │
                │         ┌─────  [1/8]  ─────  [1/8]  ────  [1/8]  ──┼→ Fuse → Output
                │         │                ↕    ↕ ↕          ↕ ↕ ↕    │
                │         │       ┌──────  [1/16] ────────  [1/16] ──┤
                │         │       │                  ↕        ↕ ↕     │
                │         │       │        ┌───────  [1/32] ─────────┘
                └─────────┴───────┴────────┴─────────────────────────┘
                
Legend: ↕ = Cross-resolution feature fusion
```

### 2.2 HRNet-W32 Specifications

| Parameter | Value |
|-----------|-------|
| **Width (W)** | 32 (base channel width at highest resolution) |
| **Stages** | 4 |
| **Resolution branches** | 4 (1/4, 1/8, 1/16, 1/32 of input) |
| **Channels per branch** | [32, 64, 128, 256] |
| **Total Parameters** | ~28.5M |
| **GFLOPs** | ~7.1 (at 256×192 input) |

### 2.3 Output Strategy for Keypoint Heatmaps

**For pose estimation (our use case):**
- Use only the **highest resolution branch** (1/4 input size = 128×128)
- Apply 1×1 conv to get 40 channels (10 vertebrae × 4 corners)
- Bilinear upsample to 512×512 to match our target heatmap size

```
HRNet-W32 Output:
    Stage 4 highest-resolution feature (128×128×32)
        ↓
    1×1 Conv → 40 channels (128×128×40)
        ↓
    Bilinear Upsample 4× → (512×512×40)
        ↓
    Output Heatmaps (512×512×40)
```

---

## 3. Implementation Plan

### 3.1 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `models/hrnet.py` | **CREATE** | HRNet backbone implementation |
| `models/hrnet_heatmap.py` | **CREATE** | HRNet + heatmap head for our task |
| `train_hrnet.py` | **CREATE** | Training script for HRNet |
| `scripts/evaluate_test.py` | **MODIFY** | Add HRNet model type support |
| `models/__init__.py` | **MODIFY** | Export HRNet model |
| `experiments/configs/hrnet_config.yaml` | **CREATE** | HRNet hyperparameters |

### 3.2 Implementation Options

**Option A: Use Official HRNet Implementation**
```bash
# Clone from official repo
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
```
- Pros: Tested, exact paper implementation
- Cons: Complex dependencies, needs adaptation

**Option B: Use timm/mmpose HRNet** ✅ Recommended
```python
import timm
model = timm.create_model('hrnet_w32', pretrained=True)
```
- Pros: Clean API, pretrained weights, easy integration
- Cons: May need to extract intermediate features

**Option C: Implement from Scratch**
- Pros: Full control, educational
- Cons: Time-consuming, potential bugs

### 3.3 Recommended Approach: timm + Custom Head

```python
import timm
import torch
import torch.nn as nn

class HRNetHeatmap(nn.Module):
    """HRNet-W32 with heatmap regression head for vertebra keypoints."""
    
    def __init__(
        self,
        num_keypoints: int = 40,  # 10 vertebrae × 4 corners
        pretrained: bool = True,
        output_size: int = 512
    ):
        super().__init__()
        
        # Load pretrained HRNet-W32 backbone
        self.backbone = timm.create_model(
            'hrnet_w32',
            pretrained=pretrained,
            features_only=True,  # Get intermediate features
            out_indices=(0,)     # Only highest resolution
        )
        
        # Heatmap head
        # HRNet-W32 outputs 32 channels at highest resolution
        self.head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        
        self.output_size = output_size
        
    def forward(self, x):
        # x: [B, 3, 512, 512]
        
        # Extract features (highest resolution: 1/4 input = 128×128)
        features = self.backbone(x)[0]  # [B, 32, 128, 128]
        
        # Generate heatmaps
        heatmaps = self.head(features)  # [B, 40, 128, 128]
        
        # Upsample to target size
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 40, 512, 512]
        
        return heatmaps
```

---

## 4. Training Configuration

### 4.1 Hyperparameters

Based on HRNet pose estimation papers and our dataset characteristics:

```yaml
# experiments/configs/hrnet_config.yaml
model:
  name: hrnet_w32
  pretrained: true
  num_keypoints: 40
  output_size: 512
  
  # Backbone freezing strategy
  freeze_backbone: false  # Train full model
  freeze_stages: 0        # Or freeze first N stages

training:
  epochs: 100
  batch_size: 4           # Same as ResNet-50
  
  # Optimizer
  optimizer: adamw
  learning_rate: 0.0001   # Lower than UNet (pretrained weights)
  weight_decay: 0.01
  
  # Scheduler
  scheduler: cosine
  warmup_epochs: 5
  min_lr: 0.000001
  
  # Loss
  loss: mse               # MSE for heatmaps
  
  # Augmentation
  augmentation:
    horizontal_flip: false  # Anatomical laterality matters
    rotation_range: 10
    scale_range: [0.9, 1.1]
    brightness: 0.1
    contrast: 0.1

data:
  train_split: 0.85
  val_split: 0.15
  image_size: 512
  heatmap_size: 512
  sigma: 6.0              # Gaussian sigma for heatmaps

evaluation:
  save_best: both         # Save best loss AND best MRE
  sdr_thresholds: [4, 8, 12, 16, 20, 24]
```

### 4.2 Training Strategy

| Phase | Epochs | LR | Backbone | Notes |
|-------|--------|-----|----------|-------|
| **Phase 1** | 1-20 | 1e-4 | Frozen | Warm up head only |
| **Phase 2** | 21-100 | 1e-4 → 1e-6 | Unfrozen | Fine-tune full model |

**Alternative: Full Training from Start**
- Since we have pretrained weights, can train everything with low LR
- Monitor for overfitting given small dataset (~264 images)

---

## 5. Expected Results

### 5.1 Performance Targets

Based on HRNet's proven superiority over ResNet for keypoint detection:

| Metric | UNet | ResNet-50 | HRNet-W32 (Target) |
|--------|------|-----------|-------------------|
| **MRE** | 65.07 px | 51.06 px | **<45 px** (~10% improvement) |
| **SDR@24px** | 29.7% | 36.1% | **>40%** |
| **Training Time** | 5.86 hrs | 7.90 hrs | ~7-9 hrs |

### 5.2 Why We Expect Improvement

1. **Multi-resolution features**: Maintains spatial precision at all scales
2. **Better gradient flow**: Parallel branches prevent vanishing gradients
3. **Proven on similar tasks**: COCO pose estimation (74.4% AP vs ResNet-152's 72.0%)
4. **Right capacity**: Not too large to overfit, not too small to underfit

---

## 6. Implementation Phases

### Phase 1: Setup (1-2 hours)
- [ ] Install dependencies (`timm` if not present)
- [ ] Verify HRNet pretrained weights download
- [ ] Create `models/hrnet_heatmap.py`
- [ ] Test model instantiation and forward pass

### Phase 2: Training Script (1-2 hours)
- [ ] Create `train_hrnet.py` based on `train_resnet.py`
- [ ] Create `experiments/configs/hrnet_config.yaml`
- [ ] Test training loop with 1 epoch

### Phase 3: Training (8-10 hours)
- [ ] Run full training (100 epochs)
- [ ] Monitor training curves
- [ ] Save checkpoints (best loss + best MRE)

### Phase 4: Evaluation (1 hour)
- [ ] Update `scripts/evaluate_test.py` for HRNet
- [ ] Run test set evaluation
- [ ] Generate visualizations
- [ ] Compare with UNet and ResNet-50

### Phase 5: Documentation (30 min)
- [ ] Update results in this document
- [ ] Update ADVISOR_MEETING_SUMMARY.md
- [ ] Commit and push

---

## 7. Potential Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **timm feature extraction** | Use `features_only=True` with `out_indices` |
| **Memory issues** | Reduce batch size to 2 if needed |
| **Overfitting** | Add dropout, stronger augmentation, early stopping |
| **Slow convergence** | Use warmup, check learning rate |
| **Feature resolution mismatch** | Verify HRNet output is 1/4 input size, adjust accordingly |

---

## 8. Code Checklist

### 8.1 Model File (`models/hrnet_heatmap.py`)

```python
"""
HRNet-W32 Heatmap Model for Vertebra Keypoint Detection

Key components:
1. HRNet-W32 backbone (pretrained on ImageNet)
2. Simple convolutional heatmap head
3. Bilinear upsampling to target resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional

class HRNetHeatmap(nn.Module):
    # ... implementation
```

### 8.2 Training Script (`train_hrnet.py`)

```python
"""
Training script for HRNet-W32 vertebra keypoint detection.

Usage:
    python train_hrnet.py --config experiments/configs/hrnet_config.yaml
"""
# ... implementation based on train_resnet.py
```

### 8.3 Config File (`experiments/configs/hrnet_config.yaml`)

```yaml
# HRNet-W32 configuration for vertebra keypoint detection
# ... as specified in section 4.1
```

---

## 9. References

1. **HRNet Paper**: Sun, K., et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." CVPR 2019.
   - [Paper](https://arxiv.org/abs/1902.09212)
   - [Official Code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

2. **HRNet TPAMI**: Wang, J., et al. "Deep High-Resolution Representation Learning for Visual Recognition." TPAMI 2020.
   - [Paper](https://arxiv.org/abs/1908.07919)

3. **timm Library**: 
   - [Documentation](https://huggingface.co/docs/timm/main/en/models/hrnet)
   - [HRNet Models](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hrnet.py)

4. **SimpleBaseline** (related work):
   - Xiao, B., et al. "Simple Baselines for Human Pose Estimation and Tracking." ECCV 2018.

---

## 10. Actual Results

✅ **Training completed January 26, 2026**

### 10.1 Training Results

| Metric | Value | Epoch |
|--------|-------|-------|
| Best Val Loss | **0.3753** | 87 |
| Best Val MRE | **43.85 px** | 87 |
| Best Val MSE | 3646.54 px² | 87 |
| Best Val SDR@6px | 26.14% | 87 |
| Best Val SDR@12px | 38.82% | 87 |
| Best Val SDR@18px | 42.28% | 87 |
| Best Val SDR@24px | **43.65%** | 87 |
| Training Time | 4.71 hours | - |

### 10.2 Model Comparison (Validation Set)

| Metric | HRNet-W32 | ResNet-50 | UNet | Improvement (vs ResNet) |
|--------|-----------|-----------|------|-------------------------|
| **Val MRE** | **43.85 px** | 51.06 px | 48.41 px | **-14.1%** ✅ |
| **Val SDR@24px** | **43.65%** | 36.1% | 38.0% | **+7.5%** ✅ |
| Val SDR@18px | 42.28% | - | 36.4% | - |
| Val SDR@12px | 38.82% | - | 31.9% | - |
| Val SDR@6px | 26.14% | - | 19.5% | - |
| Parameters | 31.78M | 30.79M | 8.65M | - |
| Training Time | 4.71 hr | ~6 hr | ~4 hr | - |

### 10.3 Key Observations

1. **HRNet-W32 is the best model so far** - Lowest MRE and highest SDR@24px
2. **High-resolution features pay off** - HRNet's parallel multi-resolution branches maintain spatial precision
3. **Consistent improvement across all SDR thresholds** - Shows better localization at all precision levels
4. **Faster training** - Completed in 4.71 hours vs ~6 hours for ResNet-50
5. **Converged around epoch 87** - Good convergence with cosine scheduler

### 10.4 Model Checkpoint Location

```
experiments/results/hrnet/hrnet_w32_pretrained_ss_20260126_123737/
├── best_model.pth
├── config.json
├── training_history.json
└── checkpoint_epoch_*.pth
```

---

## 11. Status

| Phase | Status | Date |
|-------|--------|------|
| Phase 1: Setup | ✅ Complete | Jan 26, 2026 |
| Phase 2: Training Script | ✅ Complete | Jan 26, 2026 |
| Phase 3: Training | ✅ Complete | Jan 26, 2026 |
| Phase 4: Evaluation | ✅ Complete | Jan 26, 2026 |
| Phase 5: Documentation | ✅ Complete | Jan 26, 2026 |

---

*Document created: January 26, 2026*
*Last updated: January 26, 2026*
