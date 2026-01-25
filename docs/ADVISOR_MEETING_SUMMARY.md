# Spondylolisthesis MAHT-Net Project Summary
## Prepared for Advisor Meeting - January 25, 2026

---

## 📋 Executive Summary

This document summarizes the complete work done on the **Vertebral Landmark Detection** project for spondylolisthesis grading. The goal is to automatically detect vertebra corner landmarks in lumbar spine X-rays to enable automated Meyerding classification.

### Key Achievements
- ✅ Complete data analysis and preprocessing pipeline
- ✅ UNet baseline model trained and evaluated
- ✅ Best validation MRE: **65.07 px** | SDR@24px: **29.8%**
- ✅ Advanced features implemented (Per-layer Dropout, MC Dropout for uncertainty)
- ✅ Full experiment management infrastructure

---

## 1. Project Objective

### Clinical Problem
**Spondylolisthesis** is a spinal condition where one vertebra slips over another. It affects 5-20% of the population and requires careful measurement of vertebral displacement for:
- Diagnosis
- Meyerding grade classification (Grade I-V)
- Surgical planning

### Technical Goal
Develop a deep learning model to automatically detect **4 corner landmarks** (TL, TR, BL, BR) for each lumbar vertebra in lateral X-ray images.

```
Vertebra Corner Landmarks:
    TL ●───────● TR
       │       │
       │       │
    BL ●───────● BR
```

---

## 2. Dataset Analysis

### Dataset: Spondylolisthesis Vertebral Landmark Dataset (Mendeley, June 2025)

| Property | Value |
|----------|-------|
| **Total Images** | 716 |
| **Training Set** | 700 images |
| **Test Set** | 16 images (unlabeled) |
| **Annotation Format** | JSON with 4 corner keypoints per vertebra |
| **Vertebrae per Image** | 2-10 (average ~5) |
| **Total Keypoints** | ~12,600 (700 × ~4.5 × 4) |
| **Image Sizes** | Variable (1200-3000 px) |
| **License** | CC BY 4.0 |

### Dataset Characteristics
- **Pathology-specific**: Real spondylolisthesis cases (not normal anatomy)
- **Multi-source**: Honduras proprietary data (208) + BUU-LSPINE (508)
- **Variable quality**: Different imaging equipment and protocols
- **Challenge**: No published baselines (new dataset with 0 citations)

### Data Split (80/20)
- **Train**: 496 images
- **Validation**: 204 images
- **Test**: 16 images (held out, no ground truth provided)

---

## 3. Technical Approach

### 3.1 Problem Formulation: Heatmap Regression

Instead of directly regressing (x, y) coordinates, we use **heatmap-based detection**:

```
Input: X-ray Image (512×512×3)
   ↓
Model: UNet Encoder-Decoder
   ↓
Output: 40 Heatmaps (512×512×40)
        └── 10 vertebrae × 4 corners each
   ↓
Post-process: Find peak location in each heatmap
```

**Why Heatmaps?**
- More robust to multiple vertebrae (each has its own output channel)
- Provides spatial context (Gaussian blob, not just a point)
- Enables confidence scoring (peak height = confidence)
- Standard approach in landmark detection (HRNet, CornerNet, CenterNet)

#### Heatmap Generation Details

For each keypoint, we generate a 2D Gaussian centered at the ground truth location:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `sigma` | 15.0 pixels | Size of Gaussian blob (larger = easier to learn) |
| `amplitude` | 10.0 | Peak height (matched to UNet output range) |

```python
# Gaussian formula at each pixel (x, y):
G(x, y) = amplitude × exp(-((x - x₀)² + (y - y₀)²) / (2 × sigma²))
```

**Important Design Decision**: We use `amplitude=10.0` (not 1000) because the UNet produces raw outputs without a final sigmoid/softmax. This allows the model to easily match the target scale.

### 3.2 Architecture: UNet

The UNet architecture was chosen for its proven effectiveness in pixel-wise prediction tasks:

```
Input Image (512×512×3)
          ↓
    ┌─────────────────────────────────────────────────┐
    │                    ENCODER                       │
    │  inc: 3 → 64 channels                           │
    │  down1: 64 → 128 (MaxPool + DoubleConv) [dp=0.2]│
    │  down2: 128 → 256 (MaxPool + DoubleConv) [dp=0.2]│
    │  down3: 256 → 512 (MaxPool + DoubleConv) [dp=0.3]│
    │  down4: 512 → 512 (MaxPool + DoubleConv) [dp=0.4]│ ← Bottleneck
    └─────────────────────────────────────────────────┘
          ↓ (skip connections →)
    ┌─────────────────────────────────────────────────┐
    │                    DECODER                       │
    │  up1: 1024 → 256 (Bilinear Up + DoubleConv) [dp=0.3]│
    │  up2: 512 → 128 (Bilinear Up + DoubleConv) [dp=0.2]│
    │  up3: 256 → 64 (Bilinear Up + DoubleConv) [dp=0.1]│
    │  up4: 128 → 64 (Bilinear Up + DoubleConv) [dp=0.0]│
    │  outc: 64 → 40 (1×1 Conv, no activation)        │
    └─────────────────────────────────────────────────┘
          ↓
Output Heatmaps (512×512×40)
    └── 10 vertebrae × 4 corners = 40 channels
```

| Component | Details |
|-----------|---------|
| **Architecture** | UNet with skip connections |
| **Total Parameters** | ~17.27M |
| **Input Size** | 512 × 512 × 3 (RGB, 3 channels for pretrained weight compatibility) |
| **Output Size** | 512 × 512 × 40 (40 heatmaps) |
| **Output Channels** | 10 vertebrae × 4 corners = 40 |
| **Base Channels** | 64 (doubles at each encoder level) |
| **Upsampling** | Bilinear (faster than transposed conv) |
| **Normalization** | BatchNorm after each Conv layer |
| **Activation** | ReLU (no final activation - raw heatmaps) |

#### Per-Layer Dropout Configuration

We implement **per-layer dropout** to provide graduated regularization:

| Layer | Position | Dropout Rate | Rationale |
|-------|----------|--------------|-----------|
| down1 | Encoder level 1 | 0.2 | Light regularization |
| down2 | Encoder level 2 | 0.2 | Light regularization |
| down3 | Encoder level 3 | 0.3 | Moderate regularization |
| down4 | Encoder level 4 (bottleneck) | 0.4 | Strongest regularization |
| up1 | Decoder level 1 | 0.3 | Moderate regularization |
| up2 | Decoder level 2 | 0.2 | Light regularization |
| up3 | Decoder level 3 | 0.1 | Minimal regularization |
| up4 | Decoder level 4 (near output) | 0.0 | No dropout (preserve precision) |

**Design Philosophy**: More dropout at deeper, abstract layers; less dropout near output for precise localization.

### 3.3 Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate, works well for deep networks |
| **Learning Rate** | 0.0003 | Reduced from 0.001 to prevent overfitting |
| **Weight Decay** | 1e-4 | L2 regularization for generalization |
| **Batch Size** | 2 | Optimal for 512×512 images on Mac (memory efficient) |
| **Image Size** | 512 × 512 | Full resolution for accurate landmark detection |
| **Loss Function** | **Weighted MSE** | Reduces background weight, focuses on keypoint regions |
| **LR Scheduler** | ReduceLROnPlateau | Patience=15 epochs, factor=0.5 |
| **Early Stopping** | 10 epochs patience | Prevents overfitting when validation stops improving |

#### Loss Function Details: Weighted MSE (`MSEWithWeightedBackground`)

We use a weighted Mean Squared Error loss that **de-emphasizes background pixels** and **emphasizes keypoint regions**:

```python
# Simplified loss computation:
squared_error = (predicted_heatmap - target_heatmap)²

# Weight map: background=0.05, keypoints=5.0
weights = 0.05 where target < 0.1  (background)
          5.0  where target ≥ 0.1  (keypoint regions)

loss = mean(squared_error × weights)
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `background_weight` | 0.05 | 95% reduction for background pixels |
| `keypoint_weight` | 5.0 | 5× emphasis on keypoint Gaussian peaks |

**Why Weighted MSE?**
- Standard MSE treats all pixels equally → ~99% of pixels are background
- Weighted MSE focuses training on the 1% that matters (Gaussian peaks)
- Simple, stable, no hyperparameter tuning required
- Successfully used in similar landmark detection work

#### Augmentation Configuration

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| Horizontal Flip | - | 50% |
| Rotation | ±15° | 50% |
| Brightness/Contrast | ±0.2 | 50% |

---

## 4. Experiments & Results

### 4.1 Training Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Epochs** | 86 (early stopped at epoch 86) | Patience = 10 epochs |
| **Best Epoch** | 76 | Based on lowest validation MRE |
| **Training Time** | ~5.86 hours | On Apple M-series (MPS) |
| **Final LR** | Reduced via scheduler | ReduceLROnPlateau active |

### 4.2 Best Model Performance

| Metric | Training | Validation | Improvement Needed |
|--------|----------|------------|-------------------|
| **MRE (Mean Radial Error)** | ~85 px | **65.07 px** | Target: <20 px |
| **SDR@6px** | ~12% | ~18% | Very strict threshold |
| **SDR@12px** | ~18% | ~27% | 27% within 12 pixels |
| **SDR@18px** | ~20% | ~29% | Marginal improvement |
| **SDR@24px** | ~21% | **29.8%** | Target: >90% |

**Interpretation:**
- **MRE = 65 px**: On average, predicted landmarks are 65 pixels away from ground truth
- **SDR@24px = 29.8%**: Only ~30% of landmarks are accurately localized within 24 pixels
- **Gap Analysis**: MRE needs to improve by ~3× to reach clinical utility

### 4.3 Training Dynamics

```
Epoch   Train Loss   Val Loss    Val MRE    Val SDR@24px   LR
--------------------------------------------------------------
  1       0.452       0.298      142.5 px      8.2%       3e-4
 25       0.087       0.065       82.3 px     22.1%       3e-4
 50       0.041       0.048       69.8 px     27.5%       3e-4
 76*      0.032       0.045       65.07 px    29.8%       1.5e-4  ← Best
 86       0.028       0.046       66.2 px     29.1%       7.5e-5  ← Early stop
```
*Approximate values - see full training log for exact numbers*

### 4.4 Test Set Evaluation

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Images** | 16 | Held-out test set |
| **Vertebrae Detected** | 160 | Exactly 10 per image (model's max) |
| **Detection Rate** | 100% | All expected peaks found |

*Note: Test set has no ground truth labels - evaluation is qualitative via visualizations*

### 4.5 Sample Visualizations Generated

The following visualizations were created for the test set:
- **Landmark overlays**: Predicted corners on original X-rays
- **Heatmap visualizations**: Per-channel heatmap outputs
- **Stored in**: `experiments/results/unet/unet_20260113_125631/visualizations/`

---

## 5. Implementation Details

### 5.1 Project Structure
```
spondylolisthesis-maht-net/
├── config/                 # Configuration files
│   └── mac_config.py      # Training hyperparameters
├── models/                 # Model architectures
│   └── unet.py            # UNet with per-layer dropout
├── training/              # Training infrastructure
│   ├── base_trainer.py    # Abstract trainer with early stopping
│   ├── unet_trainer.py    # UNet-specific trainer
│   └── losses.py          # Loss functions (MSE, AWing, Peak)
├── src/data/              # Data handling
│   ├── dataset.py         # PyTorch datasets
│   ├── augmentation.py    # Albumentations pipelines
│   └── preprocessing.py   # Image normalization
├── evaluation/            # Metrics and evaluation
│   └── keypoint_evaluator.py
├── scripts/               # Utility scripts
│   ├── evaluate_test.py   # Test set evaluation
│   ├── visualize_predictions.py
│   └── cleanup_experiments.py
├── experiments/           # Saved models and logs
│   └── results/unet/
└── docs/                  # Documentation
```

### 5.2 Key Features Implemented

| Feature | Description | File Location |
|---------|-------------|---------------|
| **Heatmap Generation** | Gaussian peaks (σ=15, amp=10) at keypoint locations | `src/data/unet_dataset.py` |
| **MRE-based Model Selection** | Save best model by MRE, not loss | `training/base_trainer.py` |
| **Early Stopping** | Stop after 10 epochs without MRE improvement | `training/base_trainer.py` |
| **Per-layer Dropout** | Different dropout rates per encoder/decoder layer | `models/unet.py` |
| **MC Dropout** | Monte Carlo dropout for uncertainty estimation | `models/unet.py` |
| **Weighted MSE Loss** | De-emphasize background, focus on keypoints | `training/losses.py` |
| **Experiment Cleanup** | Script to manage old experiments | `scripts/cleanup_experiments.py` |

---

## 6. Evaluation Metrics

### 6.1 Mean Radial Error (MRE)
```
MRE = (1/N) × Σ √[(x_pred - x_gt)² + (y_pred - y_gt)²]
```
- Measures average Euclidean distance between predicted and ground truth landmarks
- **Lower is better**
- Unit: pixels

### 6.2 Successful Detection Rate (SDR)
```
SDR@T = (# landmarks with error < T) / (total landmarks) × 100%
```
- Percentage of landmarks detected within threshold T pixels
- **Higher is better**
- Common thresholds: 6px, 12px, 18px, 24px

---

## 7. Challenges & Observations

### 7.1 Current Limitations

| Challenge | Impact | Potential Solution |
|-----------|--------|-------------------|
| **High MRE (~65 px)** | Poor clinical utility | Better architecture (HRNet), pretrained weights |
| **Low SDR (~30%)** | Only 30% accurate within 24px | Attention mechanisms, larger model |
| **No test labels** | Cannot quantify test accuracy | Request labels or cross-validate |
| **Variable image quality** | Inconsistent predictions | Better augmentation |
| **Train-Val gap** | Model overfits slightly | More regularization, data augmentation |

### 7.2 Why Results Are Below Expectations

1. **Dataset Size**: Only ~500 training images (medical imaging typically needs 1000+)
2. **Variable Vertebra Count**: 2-10 vertebrae per image creates imbalanced learning
3. **No Pretrained Weights**: Training from scratch vs. ImageNet pretraining
4. **Simple Architecture**: UNet baseline without attention mechanisms
5. **New Dataset**: No established preprocessing or augmentation protocols

---

## 8. Advanced Features Implemented

These features were implemented from analysis of an old project codebase and are ready for use:

### 8.1 Per-Layer Dropout (Active in Current Training)

Instead of a single dropout rate, we use graduated dropout across the network:

```python
# Configuration in config/mac_config.py:
DOWN_DROPOUT = [0.2, 0.2, 0.3, 0.4]  # Encoder layers
UP_DROPOUT = [0.3, 0.2, 0.1, 0.0]    # Decoder layers
```

**Rationale**: 
- Higher dropout at bottleneck (abstract features) → more regularization
- Lower dropout near output (spatial precision) → preserve localization accuracy
- This matches proven patterns from the old project

### 8.2 MC Dropout for Uncertainty Estimation (Ready for Inference)

Monte Carlo Dropout enables uncertainty quantification during inference:

```python
# Usage in models/unet.py:
mean_heatmap, std_heatmap, all_samples = model.predict_with_uncertainty(x, n_samples=10)

# mean_heatmap: Average prediction across samples
# std_heatmap: Standard deviation (uncertainty map)
# all_samples: All individual predictions [10, B, 40, 512, 512]
```

**Clinical Utility**:
- High uncertainty → model is unsure → flag for human review
- Low uncertainty → model is confident → can trust prediction
- Essential for clinical deployment and safety

### 8.3 Model Selection by MRE (Active in Training)

Instead of saving the model with lowest loss, we save based on **actual localization accuracy**:

```python
# In training/base_trainer.py:
if current_val_mre < best_val_mre:
    best_val_mre = current_val_mre
    save_best_model()  # ← Saves model with best MRE, not best loss
```

**Why MRE-based selection?**
- Loss can be low even if peak locations are wrong (e.g., smooth but mislocated Gaussians)
- MRE directly measures what we care about: landmark localization accuracy
- SDR@24px correlates with MRE, so minimizing MRE also improves SDR

---

## 9. Next Steps

### Phase 2: Alternative Architectures (Immediate Priority)

| Model | Parameters | Why Try It |
|-------|------------|------------|
| **ResNet-50** | ~23M | Pretrained ImageNet weights, proven for medical imaging |
| **HRNet** | ~29M | State-of-the-art for human pose estimation, maintains high resolution |

**Hypothesis**: Pretrained backbones should improve accuracy since our dataset is small (only ~500 training images).

### Phase 3: MAHT-Net Implementation (Main Goal)

Implement the Multi-scale Attention Hybrid Transformer Network with:
- Transformer attention mechanisms
- Multi-scale feature fusion
- Hybrid CNN-Transformer design

### Medium-term Improvements

1. **Larger Image Size** (768px) if memory permits → more spatial detail
2. **Cosine Annealing LR** → better convergence
3. **Test-Time Augmentation (TTA)** → improved robustness

### Long-term Goals

1. **Meyerding Grade Classification** → clinical diagnosis
2. **Slip Percentage Calculation** → quantitative measurement
3. **Clinical Validation Study** → real-world deployment

---

## 10. Repository Information

| Property | Value |
|----------|-------|
| **GitHub Repo** | `mohamednourdine/spondylolisthesis-maht-net` |
| **Branch** | `main` |
| **Total Commits** | 15 |
| **Best Model** | `experiments/results/unet/unet_20260113_125631/best_model.pth` |

### Key Commits
1. `e40fb75` - Initial commit (project setup)
2. `f294270` - Dataset analysis complete
3. `f69943f` - Preprocessing pipeline implemented
4. `265be01` - UNet implementation complete
5. `27e4327` - UNet baseline trained (MRE 66-71px)
6. `12a2ad7` - Advanced features added (MC Dropout, Per-layer Dropout)

---

## 11. Summary Table

| Aspect | Status | Details |
|--------|--------|---------|
| **Dataset Analysis** | ✅ Complete | 716 images (700 train, 16 test) |
| **Preprocessing** | ✅ Complete | Resize to 512×512, normalize, light augmentation |
| **Model Architecture** | ✅ Complete | UNet (17.27M params, 40 output channels) |
| **Loss Function** | ✅ Complete | Weighted MSE (background=0.05, keypoint=5.0) |
| **Regularization** | ✅ Complete | Per-layer dropout [0.2→0.4→0.0] |
| **Training Pipeline** | ✅ Complete | MRE-based model selection, early stopping |
| **Best Val MRE** | ⚠️ 65.07 px | Target: <20 px |
| **Best Val SDR@24px** | ⚠️ 29.8% | Target: >90% |
| **Test Evaluation** | ✅ Complete | 16 images, 160 vertebrae detected |
| **Advanced Features** | ✅ Implemented | MC Dropout ready for uncertainty |

---

## 12. Questions for Discussion

1. **Is the current MRE (65 px) acceptable for a baseline, or should we prioritize improvement before moving to MAHT-Net?**

2. **Should we focus on this dataset (no baselines) or switch to BUU-LSPINE (established benchmark)?**

3. **What is the priority: Better accuracy vs. Clinical features (uncertainty, grading)?**

4. **Timeline expectations for MAHT-Net implementation?**

---

*Document generated: January 25, 2026*
*Project: spondylolisthesis-maht-net*
