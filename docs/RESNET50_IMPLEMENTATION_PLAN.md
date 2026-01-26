# ResNet-50 Baseline Implementation Plan

## 📋 Overview

This document outlines the implementation plan for adding ResNet-50 as an alternative backbone for vertebral landmark detection. The goal is to leverage **pretrained ImageNet weights** to potentially improve accuracy over our UNet baseline.

---

## 1. Current State Analysis

### 1.1 UNet Baseline Results (for comparison)

| Metric | Value |
|--------|-------|
| **Best Val MRE** | 65.07 px |
| **Best Val SDR@24px** | 29.8% |
| **Parameters** | 17.27M |
| **Pretrained** | No (trained from scratch) |
| **Training Time** | ~5.86 hours (86 epochs) |

### 1.2 Existing ResNet Code

The current `models/resnet_keypoint.py` uses **direct coordinate regression**:
```python
# Current (NOT what we want):
output = model(image)  # → [B, num_keypoints, 2] (x, y coordinates)
```

**Problem**: Direct regression doesn't match our heatmap-based pipeline.

---

## 2. Proposed Architecture: ResNet-50 + Simple Decoder

### 2.1 Design Choice: Heatmap Regression (like UNet)

To maintain consistency with our evaluation pipeline and enable fair comparison.

**Key Principle**: Keep it simple! Multi-scale features and attention will be added in MAHT-Net.

```
Input Image (512×512×3)
        ↓
┌───────────────────────────────────────────────────────┐
│              ResNet-50 Backbone (Pretrained)          │
│  Conv1 → BN → ReLU → MaxPool                          │
│  Layer1 (256ch) → Layer2 (512ch) → Layer3 (1024ch)    │
│  Layer4 (2048ch) at 16×16 resolution                  │
└───────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────┐
│           Simple Upsampling Decoder                    │
│  (Based on Microsoft SimpleBaseline, 2018)            │
│                                                        │
│  DeconvBlock: 2048 → 256 (32×32)                      │
│  DeconvBlock: 256 → 256 (64×64)                       │
│  DeconvBlock: 256 → 256 (128×128)                     │
│  DeconvBlock: 256 → 256 (256×256)                     │
│  DeconvBlock: 256 → 40 (512×512)                      │
│                                                        │
│  Each DeconvBlock: ConvTranspose2d → BN → ReLU        │
└───────────────────────────────────────────────────────┘
        ↓
Output Heatmaps (512×512×40)
    └── 10 vertebrae × 4 corners = 40 channels
```

### 2.2 Why Simple Decoder (Not FPN)?

| Decision | Rationale |
|----------|-----------|
| **No FPN** | Multi-scale features reserved for MAHT-Net |
| **No attention** | Attention mechanisms reserved for MAHT-Net |
| **Simple deconv** | Proven effective (SimpleBaseline paper achieved SOTA) |
| **Fair comparison** | Isolates "pretrained weights" as the only variable vs UNet |

### 2.3 Parameter Comparison

| Component | Parameters | Notes |
|-----------|------------|-------|
| ResNet-50 backbone | ~23.5M | Pretrained weights |
| Simple decoder | ~1.5M | Trainable (5 deconv blocks) |
| **Total** | **~25M** | vs UNet's 17.27M |

---

## 3. Implementation Details

### 3.1 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `models/resnet_heatmap.py` | **CREATE** | New ResNet-50 + Simple Decoder model |
| `config/mac_config.py` | **MODIFY** | Add ResNet-specific config options |
| `train_resnet.py` | **CREATE** | Training script for ResNet-50 |
| `models/model_registry.py` | **MODIFY** | Register ResNet model |

### 3.2 Model Architecture Code (Proposed)

```python
class ResNetHeatmap(nn.Module):
    """
    ResNet-50 backbone with simple deconv decoder for heatmap regression.
    Based on SimpleBaseline (Microsoft, ECCV 2018).
    
    Architecture:
        - ResNet-50 encoder (pretrained on ImageNet)
        - Simple upsampling decoder (deconv blocks)
        - Output: heatmaps at full resolution
    
    Note: No FPN or attention - those are reserved for MAHT-Net.
    """
    
    def __init__(
        self,
        in_channels=3,
        num_keypoints=40,
        pretrained=True,
        freeze_backbone_layers=2,  # Freeze conv1, layer1
        dropout_rate=0.3
    ):
        super().__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Remove FC and avgpool (we need spatial features)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # Freeze early layers
        self._freeze_layers(freeze_backbone_layers)
        
        # Simple decoder: 5 deconv blocks to go from 16×16 to 512×512
        self.decoder = nn.Sequential(
            self._make_deconv_layer(2048, 256),  # 16→32
            self._make_deconv_layer(256, 256),   # 32→64
            self._make_deconv_layer(256, 256),   # 64→128
            self._make_deconv_layer(256, 256),   # 128→256
            self._make_deconv_layer(256, 256),   # 256→512
        )
        
        # Final conv to get heatmaps
        self.head = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def _make_deconv_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Backbone feature extraction (stop before avgpool)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # 2048 ch, 16×16
        
        # Decoder
        x = self.decoder(x)  # 256 ch, 512×512
        x = self.dropout(x)
        
        # Heatmap prediction
        heatmaps = self.head(x)  # 40 ch, 512×512
        
        return heatmaps
```

### 3.3 Training Configuration

| Parameter | UNet Value | ResNet-50 Value | Rationale |
|-----------|------------|-----------------|-----------|
| **Learning Rate** | 0.0003 | 0.0001 | Lower LR for pretrained weights |
| **Backbone LR** | - | 0.00001 | 10× smaller for backbone |
| **Batch Size** | 2 | 2 | Same (memory constraint) |
| **Freeze Layers** | - | conv1, layer1 | Preserve low-level features |
| **Optimizer** | Adam | AdamW | Better weight decay handling |
| **Weight Decay** | 1e-4 | 1e-4 | Same |
| **Epochs** | 50+ | 50+ | Same |
| **Loss** | weighted_mse | weighted_mse | Same for fair comparison |

### 3.4 Key Implementation Considerations

1. **Pretrained Weight Loading**
   ```python
   # Use torchvision's pretrained weights
   from torchvision.models import ResNet50_Weights
   backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
   ```

2. **Differential Learning Rates**
   ```python
   # Different LR for backbone vs decoder
   param_groups = [
       {'params': model.backbone.parameters(), 'lr': 1e-5},
       {'params': model.decoder.parameters(), 'lr': 1e-4},
       {'params': model.head.parameters(), 'lr': 1e-4},
   ]
   optimizer = torch.optim.AdamW(param_groups)
   ```

3. **Layer Freezing Strategy**
   ```python
   # Freeze conv1 and layer1 (low-level features)
   for name, param in model.backbone.named_parameters():
       if 'conv1' in name or 'bn1' in name or 'layer1' in name:
           param.requires_grad = False
   ```

4. **Output Resolution**
   - ResNet-50 output stride: 32 (1/32 of input)
   - Need to upsample back to 512×512
   - Use bilinear interpolation + conv refinement

---

## 4. Training Pipeline

### 4.1 Data Pipeline (Reuse from UNet)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Dataset | `src/data/unet_dataset.py` | None (same heatmap format) |
| Augmentation | `src/data/augmentation.py` | None |
| Preprocessing | ImageNet normalization | **NEW**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### 4.2 Training Script: `train_resnet.py`

```bash
# Usage
python train_resnet.py \
    --epochs 100 \
    --lr 0.0001 \
    --backbone-lr 0.00001 \
    --freeze-layers 2 \
    --pretrained
```

### 4.3 Evaluation (Same as UNet)

- MRE (Mean Radial Error) in pixels
- SDR@6px, SDR@12px, SDR@18px, SDR@24px
- Use existing `evaluation/keypoint_evaluator.py`

---

## 5. Expected Outcomes

### 5.1 Hypothesis

| Factor | Expected Impact |
|--------|-----------------|
| **Pretrained weights** | Better feature extraction → lower MRE |
| **Larger receptive field** | Better context understanding |
| **More parameters (38M vs 17M)** | Higher capacity, risk of overfitting |

### 5.2 Target vs Actual Metrics

| Metric | UNet Baseline | Target | **Actual** | Status |
|--------|---------------|--------|------------|--------|
| **Val MRE** | 65.07 px | <55 px | **51.06 px** | ✅ Exceeded |
| **Val SDR@24px** | 29.7% | >40% | **36.1%** | ⚠️ Close |

### 5.3 Risks and Mitigations

| Risk | Mitigation | Outcome |
|------|------------|---------|
| Overfitting | Freeze backbone, dropout | ✅ No overfitting observed |
| Longer training time | Reduce epochs if needed | ⚠️ 35% longer (acceptable) |
| Memory issues | Batch size=2 | ✅ No issues |
| Worse than UNet | Document findings | ✅ Significantly better |

---

## 6. Implementation Checklist

### Phase 1: Model Implementation
- [x] Create `models/resnet_heatmap.py` with ResNet-50 + Simple Decoder
- [x] Add deconv block helper
- [x] Implement layer freezing logic
- [x] Add model to `model_registry.py`

### Phase 2: Training Infrastructure
- [x] Create `train_resnet.py` script
- [x] Implement differential learning rates
- [x] Reuse UNet dataset (same heatmap format)

### Phase 3: Training & Evaluation
- [x] Train ResNet-50 model (100 epochs, 7.9 hours)
- [x] Monitor training curves (loss, MRE, SDR)
- [x] Compare with UNet baseline
- [x] Generate test set predictions

### Phase 4: Documentation
- [x] Document results
- [ ] Update ADVISOR_MEETING_SUMMARY.md
- [ ] Commit and tag version

---

## 7. Actual Training Results (January 26, 2026)

### 7.1 Final Comparison: UNet vs ResNet-50

| Metric | UNet (Baseline) | ResNet-50 | Improvement |
|--------|-----------------|-----------|-------------|
| **Best Val MRE** | 65.07 px | **51.06 px** | ✅ **21.5% better** |
| **Best Val SDR@6px** | 17.8% | **18.3%** | ✅ +0.5% |
| **Best Val SDR@12px** | 26.8% | **31.9%** | ✅ +5.1% |
| **Best Val SDR@18px** | 29.3% | **34.9%** | ✅ +5.6% |
| **Best Val SDR@24px** | 29.7% | **36.1%** | ✅ **+6.4%** |
| Best Epoch | 76 | 86 | - |
| Training Time | 5.86 hrs | 7.90 hrs | +35% |
| Parameters | 17.27M | 38.15M | +121% |

### 7.2 Key Observations

1. **Pretrained weights helped significantly!**
   - MRE improved by ~14 pixels (21.5% reduction)
   - SDR@24px improved by 6.4 percentage points

2. **Training dynamics**
   - Best loss at epoch 39, but best MRE at epoch 86
   - Model continued to improve on localization even after loss plateaued
   - Saved `best_model_mre.pth` for MRE-based checkpoint

3. **Trade-offs**
   - Longer training time (35% more)
   - 2.2× more parameters

### 7.3 Conclusion

**Hypothesis confirmed**: ImageNet pretrained weights provide significant improvement for vertebral landmark detection, even with a simple decoder architecture.

---

## 8. Architecture Progression (Roadmap)

```
UNet (baseline)          → ResNet-50 (pretrained)      → MAHT-Net (final)
─────────────────────────────────────────────────────────────────────────
Simple encoder-decoder     Pretrained backbone           Multi-scale attention
No pretrained weights      Simple decoder                Transformer blocks
17.27M params              ~25M params                   Hybrid CNN-Transformer
                           ↑                              ↑
                           WE ARE HERE                    ULTIMATE GOAL
```

| Feature | UNet | ResNet-50 | MAHT-Net |
|---------|------|-----------|----------|
| Pretrained backbone | ❌ | ✅ | ✅ |
| Multi-scale features | ❌ | ❌ | ✅ (FPN/etc) |
| Attention mechanisms | ❌ | ❌ | ✅ |
| Transformer blocks | ❌ | ❌ | ✅ |

---

## 8. Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Model Implementation | 2-3 hours | Core architecture |
| Training Script | 1-2 hours | Reuse UNet patterns |
| Training | 4-6 hours | ~100 epochs on Mac |
| Evaluation & Analysis | 1-2 hours | Compare with UNet |
| **Total** | **8-13 hours** | Spread over 1-2 days |

---

*Document created: January 25, 2026*
*Project: spondylolisthesis-maht-net*
*Status: Ready for Review*
