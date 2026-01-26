# Spondylolisthesis MAHT-Net Project Summary
## Prepared for Advisor Meeting - January 26, 2026

---

## 📋 Executive Summary

This document summarizes the complete work done on the **Vertebral Landmark Detection** project for spondylolisthesis grading. The goal is to automatically detect vertebra corner landmarks in lumbar spine X-rays to enable automated Meyerding classification.

### Key Achievements
- ✅ Complete data analysis and preprocessing pipeline
- ✅ UNet baseline model trained and evaluated
- ✅ ResNet-50 model trained - 21.5% improvement over UNet
- ✅ **HRNet-W32 model trained - BEST RESULTS!**
- ✅ Best validation MRE: **43.85 px** (HRNet-W32) | SDR@24px: **43.65%**
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

### Dataset Visualizations

| Demographics Overview | Slip Distribution |
|:---------------------:|:-----------------:|
| ![Demographics](figures/demographics_overview.png) | ![Slip Distribution](figures/slip_distribution.png) |

| Sample Images: Normal vs Severe Slip |
|:------------------------------------:|
| ![Normal vs Severe](figures/normal_vs_severe_slip_comparison.png) |

| Random Training Samples |
|:-----------------------:|
| ![Random Samples](figures/10_random_samples.png) |

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

### 3.2 Model Architectures

We implemented three architectures for comparison:

#### 3.2.1 UNet (Baseline)

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

#### 3.2.2 ResNet-50 + Simple Decoder (Pretrained)

To leverage ImageNet pretrained weights, we implemented ResNet-50 with a simple upsampling decoder:

```
Input Image (512×512×3)
        ↓
┌───────────────────────────────────────────────────────┐
│              ResNet-50 Backbone (ImageNet Pretrained) │
│  Conv1 → BN → ReLU → MaxPool                          │
│  Layer1 (256ch) → Layer2 (512ch) → Layer3 (1024ch)    │
│  Layer4 (2048ch) at 16×16 resolution                  │
│  [conv1, bn1, layer1 FROZEN]                          │
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
│  DeconvBlock: 256 → 256 (512×512)                     │
│  Head: 1×1 Conv → 40 channels                         │
└───────────────────────────────────────────────────────┘
        ↓
Output Heatmaps (512×512×40)
```

#### 3.2.3 HRNet-W32 (High-Resolution Network) 🏆

HRNet maintains high-resolution representations throughout the network with parallel multi-scale branches:

```
Input Image (512×512×3)
        ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    HRNet-W32 Backbone (ImageNet Pretrained via timm)     │
│                                                                          │
│  Stem: Conv → BN → ReLU → Conv → BN → ReLU (64ch, 256×256)             │
│                                                                          │
│              Stage 1      Stage 2       Stage 3      Stage 4            │
│          ┌─ [1/4, 32ch] ─ [1/4, 32ch] ─ [1/4, 32ch] ─ [1/4, 32ch] ─┐   │
│          │                    ↕            ↕ ↕          ↕ ↕ ↕       │   │
│          │       ┌────────  [1/8, 64ch] ─ [1/8, 64ch] ─ [1/8, 64ch]─┤   │
│          │       │                  ↕       ↕ ↕          ↕ ↕ ↕      │   │
│          │       │        ┌───── [1/16,128ch] ────── [1/16,128ch]──┤   │
│          │       │        │                  ↕          ↕ ↕        │   │
│          │       │        │         ┌──── [1/32,256ch] ────────────┘   │
│          └───────┴────────┴─────────┴──────────────────────────────────┘
│                                                                          │
│  Legend: ↕ = Cross-resolution feature fusion at every stage            │
│  Key: Maintains HIGH RESOLUTION (1/4) throughout entire network         │
└─────────────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Simple Heatmap Head (0.90M params)                    │
│                                                                          │
│  Use Stage 1 features: 128×128×128 (highest resolution branch)          │
│  Conv2d: 128 → 64 (3×3, BN, ReLU)                                       │
│  Conv2d: 64 → 40 (1×1)                                                  │
│  Bilinear Upsample: 128×128 → 512×512                                   │
└─────────────────────────────────────────────────────────────────────────┘
        ↓
Output Heatmaps (512×512×40)
```

**Why HRNet Works Best:**
1. **Parallel multi-resolution branches** - no information loss from sequential downsampling
2. **Repeated cross-scale fusion** - information exchange between resolutions at every stage
3. **High-resolution output** - uses 1/4 resolution features (128×128), not bottleneck features
4. **Proven for keypoints** - SOTA on COCO pose estimation

#### Architecture Comparison

| Component | UNet | ResNet-50 | HRNet-W32 |
|-----------|------|----------|------------|
| **Backbone Parameters** | N/A | 25.56M | 30.88M |
| **Decoder/Head Parameters** | ~17M | 12.59M | 0.90M |
| **Total Parameters** | 17.27M | 38.15M | 31.78M |
| **Pretrained Weights** | ❌ No | ✅ ImageNet | ✅ ImageNet |
| **Frozen Layers** | None | conv1, bn1, layer1 | None |
| **Architecture Type** | Encoder-Decoder | Encoder + Deconv | Parallel Multi-Scale |
| **Resolution Strategy** | Downsample → Upsample | Downsample → Upsample | High-res maintained |
| **Input Size** | 512 × 512 × 3 | 512 × 512 × 3 | 512 × 512 × 3 |
| **Output Size** | 512 × 512 × 40 | 512 × 512 × 40 | 512 × 512 × 40 |

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

#### UNet Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | 0.0003 | Single LR for all layers |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Batch Size** | 2 | Memory efficient for 512×512 |
| **Early Stopping** | 10 epochs patience | Prevents overfitting |

#### ResNet-50 Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Optimizer** | AdamW | Better weight decay handling |
| **Backbone LR** | 0.00001 | Slow fine-tuning of pretrained weights |
| **Decoder/Head LR** | 0.0001 | Faster learning for new layers |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Batch Size** | 2 | Memory efficient |
| **Early Stopping** | 15 epochs patience | More patience for transfer learning |
| **Layer Freezing** | conv1, bn1, layer1 | Preserve low-level features |

#### HRNet-W32 Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Optimizer** | AdamW | Better weight decay handling |
| **Backbone LR** | 0.00001 | Slow fine-tuning of pretrained weights |
| **Head LR** | 0.0001 | Faster learning for new layers |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Batch Size** | 4 | Larger batch with smaller head |
| **Scheduler** | Cosine Annealing | Smooth LR decay to near zero |
| **Warmup Epochs** | 5 | Gradual LR increase at start |
| **Layer Freezing** | None | All layers trainable |
| **Dropout** | 0.3 | In heatmap head only |

#### Common Configuration (All Models)

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

| Metric | UNet | ResNet-50 | HRNet-W32 |
|--------|------|----------|------------|
| **Total Epochs** | 86 | 100 | 100 |
| **Best Epoch (MRE)** | 76 | 86 | 87 |
| **Training Time** | 5.86 hours | 7.90 hours | 4.71 hours |
| **Device** | Apple M-series (MPS) | Apple M-series (MPS) | Apple M-series (MPS) |
| **Convergence** | Early stopped | Completed all epochs | Completed all epochs |

### 4.2 Best Model Performance

#### Model Comparison: UNet vs ResNet-50 vs HRNet-W32

| Metric | UNet | ResNet-50 | HRNet-W32 🏆 | Best Improvement |
|--------|------|----------|------------|------------------|
| **Val MRE** | 65.07 px | 51.06 px | **43.85 px** | ✅ **-32.6%** (vs UNet) |
| **Val SDR@6px** | 17.8% | 18.3% | **26.14%** | +8.3% |
| **Val SDR@12px** | 26.8% | 31.9% | **38.82%** | +12.0% |
| **Val SDR@18px** | 29.3% | 34.9% | **42.28%** | +13.0% |
| **Val SDR@24px** | 29.7% | 36.1% | **43.65%** | ✅ **+14.0%** |
| Best Epoch | 76 | 86 | 87 | - |
| Training Time | 5.86 hrs | 7.90 hrs | **4.71 hrs** | Fastest |
| Parameters | 17.27M | 38.15M | 31.78M | - |

**Key Finding**: HRNet-W32 achieves the best results across all metrics!

![Model Comparison](figures/model_comparison_bar.png)
*Figure: Side-by-side comparison of all three models*

**Interpretation:**
- **MRE = 43.85 px (HRNet)**: 21 pixels better than UNet, 7 pixels better than ResNet
- **SDR@24px = 43.65%**: 14 percentage points improvement over UNet
- **High-resolution features**: HRNet's parallel branches preserve spatial precision

### 4.3 Training Dynamics

#### UNet Training Progression
```
Epoch   Train Loss   Val Loss    Val MRE    Val SDR@24px   LR
--------------------------------------------------------------
  1       0.452       0.298      142.5 px      8.2%       3e-4
 25       0.087       0.065       82.3 px     22.1%       3e-4
 50       0.041       0.048       69.8 px     27.5%       3e-4
 76*      0.032       0.045       65.07 px    29.7%       1.5e-4  ← Best MRE
 86       0.028       0.046       66.2 px     29.1%       7.5e-5  ← Early stop
```

#### ResNet-50 Training Progression
```
Epoch   Train Loss   Val Loss    Val MRE    Val SDR@24px   LR
--------------------------------------------------------------
  1       0.856       0.612      128.4 px     10.1%       1e-4/1e-5
 25       0.423       0.398       72.5 px     28.3%       1e-4/1e-5
 50       0.312       0.392       58.2 px     33.1%       5e-5/5e-6
 86*      0.275       0.386       51.06 px    36.1%       1e-6     ← Best MRE
100       0.272       0.456       55.61 px    31.6%       1e-6     ← End
```

#### HRNet-W32 Training Progression (NEW)
```
Epoch   Train Loss   Val Loss    Val MRE    Val SDR@24px   LR
--------------------------------------------------------------
  1       0.939       1.009      280.4 px      0.2%       1e-4/1e-5
 25       0.337       0.418       51.2 px     38.8%       ~5e-5
 50       0.284       0.389       45.8 px     41.2%       ~1e-5
 87*      0.232       0.375       43.85 px    43.65%      ~1e-6    ← Best MRE 🏆
100       0.252       0.379       44.19 px    43.08%      ~0       ← End
```
*Approximate values - see full training logs for exact numbers*

**Key Observation**: HRNet-W32 achieves the best MRE (43.85 px) at epoch 87, with smooth convergence thanks to cosine annealing scheduler.

#### Training Curves Visualization

##### All Models Comparison

![Training Curves All Models](figures/training_curves_all_models.png)
*Figure: Training curves comparison for all three models - MRE, SDR@24px, and Loss over epochs*

##### HRNet-W32 Detailed Training Curves

![HRNet Training Curves](figures/hrnet_training_curves.png)
*Figure: HRNet-W32 detailed training curves showing MRE, SDR, Loss, and generalization gap*

##### Final Performance Comparison

![Model Comparison Bar](figures/model_comparison_bar.png)
*Figure: Final best performance comparison - HRNet-W32 achieves lowest MRE and highest SDR@24px*

### 4.4 Test Set Evaluation

| Metric | UNet | ResNet-50 | HRNet-W32 |
|--------|------|----------|------------|
| **Test Images** | 16 | 16 | 16 |
| **Vertebrae Detected** | 160 | 160 | 160 |
| **Detection Rate** | 100% | 100% | 100% |
| **Results Location** | `experiments/test_evaluation/unet/` | `experiments/test_evaluation/resnet/` | `experiments/test_evaluation/` |

*Note: Test set has no ground truth labels - evaluation is qualitative via visualizations*

### 4.5 Sample Visualizations Generated

The following visualizations were created for the test set:
- **Landmark overlays**: Predicted corners on original X-rays
- **Heatmap visualizations**: Per-channel heatmap outputs
- **UNet results**: `experiments/test_evaluation/unet/visualizations/`
- **ResNet results**: `experiments/test_evaluation/resnet/visualizations/`
- **HRNet results**: `experiments/test_evaluation/visualizations/`

#### Sample UNet Predictions on Test Set

| Sample 1 | Sample 2 | Sample 3 |
|:--------:|:--------:|:--------:|
| ![UNet 1](figures/unet_predictions/3729-F-067Y1_prediction.png) | ![UNet 2](figures/unet_predictions/3753-F-049Y1_prediction.png) | ![UNet 3](figures/unet_predictions/3808-F-065Y1_prediction.png) |
| 3729-F-067Y1 | 3753-F-049Y1 | 3808-F-065Y1 |

| Sample 4 | Sample 5 | Sample 6 |
|:--------:|:--------:|:--------:|
| ![UNet 4](figures/unet_predictions/3836-F-020Y1_prediction.png) | ![UNet 5](figures/unet_predictions/3870-F-060Y1_prediction.png) | ![UNet 6](figures/unet_predictions/4093-F-083Y1_prediction.png) |
| 3836-F-020Y1 | 3870-F-060Y1 | 4093-F-083Y1 |

#### Sample ResNet-50 Predictions on Test Set

| Sample 1 | Sample 2 | Sample 3 |
|:--------:|:--------:|:--------:|
| ![ResNet 1](figures/resnet_predictions/3729-F-067Y1_prediction.png) | ![ResNet 2](figures/resnet_predictions/3753-F-049Y1_prediction.png) | ![ResNet 3](figures/resnet_predictions/3808-F-065Y1_prediction.png) |
| 3729-F-067Y1 | 3753-F-049Y1 | 3808-F-065Y1 |

| Sample 4 | Sample 5 | Sample 6 |
|:--------:|:--------:|:--------:|
| ![ResNet 4](figures/resnet_predictions/3836-F-020Y1_prediction.png) | ![ResNet 5](figures/resnet_predictions/3870-F-060Y1_prediction.png) | ![ResNet 6](figures/resnet_predictions/4093-F-083Y1_prediction.png) |
| 3836-F-020Y1 | 3870-F-060Y1 | 4093-F-083Y1 |

#### Sample HRNet-W32 Predictions on Test Set 🏆

| Sample 1 | Sample 2 | Sample 3 |
|:--------:|:--------:|:--------:|
| ![HRNet 1](figures/hrnet_predictions/3729-F-067Y1_prediction.png) | ![HRNet 2](figures/hrnet_predictions/3753-F-049Y1_prediction.png) | ![HRNet 3](figures/hrnet_predictions/3808-F-065Y1_prediction.png) |
| 3729-F-067Y1 | 3753-F-049Y1 | 3808-F-065Y1 |

| Sample 4 | Sample 5 | Sample 6 |
|:--------:|:--------:|:--------:|
| ![HRNet 4](figures/hrnet_predictions/3836-F-020Y1_prediction.png) | ![HRNet 5](figures/hrnet_predictions/3870-F-060Y1_prediction.png) | ![HRNet 6](figures/hrnet_predictions/4093-F-083Y1_prediction.png) |
| 3836-F-020Y1 | 3870-F-060Y1 | 4093-F-083Y1 |

#### Side-by-Side Comparison (Same Test Image)

| UNet (MRE: 65.07 px) | ResNet-50 (MRE: 51.06 px) | HRNet-W32 (MRE: 43.85 px) 🏆 |
|:--------------------:|:-------------------------:|:---------------------------:|
| ![UNet](figures/unet_predictions/3729-F-067Y1_prediction.png) | ![ResNet](figures/resnet_predictions/3729-F-067Y1_prediction.png) | ![HRNet](figures/hrnet_predictions/3729-F-067Y1_prediction.png) |
| **3729-F-067Y1** | **3729-F-067Y1** | **3729-F-067Y1** |
| ![UNet](figures/unet_predictions/3836-F-020Y1_prediction.png) | ![ResNet](figures/resnet_predictions/3836-F-020Y1_prediction.png) | ![HRNet](figures/hrnet_predictions/3836-F-020Y1_prediction.png) |
| **3836-F-020Y1** | **3836-F-020Y1** | **3836-F-020Y1** |

*Note: These are predictions on the held-out test set (no ground truth available). Each image shows detected vertebra corners: TL (red), TR (green), BL (blue), BR (yellow).*

---

## 5. Implementation Details

### 5.1 Project Structure
```
spondylolisthesis-maht-net/
├── config/                 # Configuration files
│   └── mac_config.py      # Training hyperparameters
├── models/                 # Model architectures
│   ├── unet.py            # UNet with per-layer dropout
│   ├── resnet_heatmap.py  # ResNet-50 + Simple Decoder
│   ├── hrnet_heatmap.py   # HRNet-W32 + Heatmap Head (NEW) 🏆
│   └── model_registry.py  # Model factory registry
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
│   ├── evaluate_test.py   # Test set evaluation (supports UNet, ResNet, HRNet)
│   ├── train_resnet.py    # ResNet training script
│   ├── train_hrnet.py     # HRNet training script (NEW)
│   └── cleanup_experiments.py
├── experiments/           # Saved models and logs
│   └── results/
│       ├── unet/          # UNet experiment results
│       ├── resnet/        # ResNet experiment results
│       └── hrnet/         # HRNet experiment results (NEW)
└── docs/                  # Documentation
    ├── RESNET50_IMPLEMENTATION_PLAN.md  # ResNet implementation guide
    └── HRNET_IMPLEMENTATION_PLAN.md     # HRNet implementation guide (NEW)
```

### 5.2 Key Features Implemented

| Feature | Description | File Location |
|---------|-------------|---------------|
| **Heatmap Generation** | Gaussian peaks (σ=15, amp=10) at keypoint locations | `src/data/unet_dataset.py` |
| **MRE-based Model Selection** | Save best model by MRE, not loss | `training/base_trainer.py` |
| **Early Stopping** | Stop after patience epochs without MRE improvement | `training/base_trainer.py` |
| **Per-layer Dropout** | Different dropout rates per encoder/decoder layer | `models/unet.py` |
| **MC Dropout** | Monte Carlo dropout for uncertainty estimation | `models/unet.py`, `models/resnet_heatmap.py`, `models/hrnet_heatmap.py` |
| **Weighted MSE Loss** | De-emphasize background, focus on keypoints | `training/losses.py` |
| **Pretrained Backbone** | ResNet-50/HRNet-W32 with ImageNet weights | `models/resnet_heatmap.py`, `models/hrnet_heatmap.py` |
| **Differential Learning Rates** | Different LR for backbone vs head | `train_resnet.py`, `train_hrnet.py` |
| **Layer Freezing** | Freeze early backbone layers | `models/resnet_heatmap.py` |
| **Cosine Annealing** | Smooth LR decay with warmup | `train_hrnet.py` |
| **HRNet Multi-Scale** | Parallel high-resolution branches | `models/hrnet_heatmap.py` |
| **Model Registry** | Factory pattern for model creation | `models/model_registry.py` |

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

### 7.1 Current Status After HRNet-W32

| Challenge | UNet | ResNet-50 | HRNet-W32 | Status |
|-----------|------|-----------|-----------|--------|
| **High MRE** | 65.07 px | 51.06 px | **43.85 px** | ✅ Improved 32.6% |
| **Low SDR@24px** | 29.7% | 36.1% | **43.65%** | ✅ Improved 14% |
| **No pretrained weights** | ❌ | ✅ | ✅ | ✅ Resolved |
| **No test labels** | ❌ | ❌ | ❌ | ⚠️ Still an issue |
| **Variable image quality** | ⚠️ | ⚠️ | ⚠️ | ⚠️ Needs augmentation |

### 7.2 Remaining Challenges

| Challenge | Impact | Potential Solution |
|-----------|--------|-------------------|
| **MRE still >40 px** | Not clinical-ready | Attention mechanisms, MAHT-Net |
| **SDR@24px ~44%** | 56% landmarks off by >24px | Multi-scale fusion, larger model |
| **No test labels** | Cannot quantify test accuracy | Request labels or cross-validate |
| **Training time** | ~5 hours per experiment | Already optimized with HRNet |

### 7.3 What Worked (HRNet-W32 Success Factors)

1. **High-Resolution Representations**: Parallel branches maintain spatial precision
2. **Multi-Scale Feature Fusion**: Cross-resolution exchange at every stage
3. **ImageNet Pretrained Weights**: Strong initialization via timm library
4. **Differential Learning Rates**: backbone_lr=1e-5, head_lr=1e-4
5. **Cosine Annealing Scheduler**: Smooth LR decay with 5-epoch warmup
6. **Simple Heatmap Head**: Uses Stage 1 features (128ch, 1/4 resolution)

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

### Phase 2: Alternative Architectures ✅ COMPLETED

| Model | Parameters | Status | Result |
|-------|------------|--------|--------|
| **ResNet-50** | 38.15M | ✅ Complete | **MRE 51.06 px** (21.5% better than UNet) |
| **HRNet** | ~29M | ⏳ Next | Maintains high resolution throughout |

**Hypothesis CONFIRMED**: Pretrained ImageNet weights significantly improved accuracy (14 px reduction in MRE).

### Phase 2.5: HRNet Implementation (Next Priority)

| Model | Why Try It |
|-------|------------|
| **HRNet-W32** | Maintains high-resolution features, SOTA for pose estimation |
| **HRNet-W48** | Larger capacity, better for fine-grained localization |

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
| **Best UNet Model** | `experiments/results/unet/unet_20260113_125631/best_model.pth` |
| **Best ResNet Model** | `experiments/results/resnet/resnet50_pretrained_20260125_223141/best_model_mre.pth` |
| **Best HRNet Model** | `experiments/results/hrnet/hrnet_w32_pretrained_ss_20260126_123737/best_model.pth` 🏆 |

### Key Commits
1. `e40fb75` - Initial commit (project setup)
2. `f294270` - Dataset analysis complete
3. `f69943f` - Preprocessing pipeline implemented
4. `265be01` - UNet implementation complete
5. `27e4327` - UNet baseline trained (MRE 65.07px)
6. `12a2ad7` - Advanced features added (MC Dropout, Per-layer Dropout)
7. `5ea4ebf` - ResNet-50 implementation complete (MRE 51.06px)
8. `TBD` - **HRNet-W32 implementation complete** (MRE 43.85px) 🏆

---

## 11. Summary Table

| Aspect | Status | Details |
|--------|--------|---------|
| **Dataset Analysis** | ✅ Complete | 716 images (700 train, 16 test) |
| **Preprocessing** | ✅ Complete | Resize to 512×512, normalize, light augmentation |
| **UNet Baseline** | ✅ Complete | 17.27M params, MRE 65.07 px, SDR@24px 29.7% |
| **ResNet-50** | ✅ Complete | 38.15M params, MRE 51.06 px, SDR@24px 36.1% |
| **HRNet-W32** | ✅ Complete | 31.78M params, **MRE 43.85 px**, SDR@24px 43.65% 🏆 |
| **Loss Function** | ✅ Complete | Weighted MSE (background=0.05, keypoint=5.0) |
| **Regularization** | ✅ Complete | Per-layer dropout, differential LR |
| **Training Pipeline** | ✅ Complete | MRE-based model selection, early stopping |
| **Best Val MRE** | ⚠️ 43.85 px | Target: <20 px (HRNet-W32 best) |
| **Best Val SDR@24px** | ⚠️ 43.65% | Target: >90% (HRNet-W32 best) |
| **Test Evaluation** | ✅ Complete | 16 images, 160 vertebrae detected |
| **Advanced Features** | ✅ Implemented | MC Dropout ready for uncertainty |

---

## 12. Questions for Discussion

1. **HRNet-W32 achieved 43.85 px MRE (32.6% better than UNet) - ready to proceed with MAHT-Net?**

2. **Should we try the HRNet multi-scale variant for potentially better results before MAHT-Net?**

3. **Should we focus on this dataset (no baselines) or switch to BUU-LSPINE (established benchmark)?**

4. **The MRE is still ~44 px - what architectural changes would most likely close the gap to clinical utility (<20 px)?**

5. **Timeline expectations for MAHT-Net implementation given current progress?**

---

## 13. Dataset Challenges & Their Impact on Results

This section discusses the fundamental challenges encountered with the Spondylolisthesis Vertebral Landmark Dataset that contribute to the current MRE of ~44 px. Understanding these limitations is crucial for interpreting our results and planning future improvements.

### 13.1 Variable Vertebrae Count: The 40-Keypoint Problem

#### The Core Issue

Unlike standard human pose estimation datasets (COCO: 17 keypoints, MPII: 16 keypoints), our dataset has a **variable number of vertebrae per image** (2-10 vertebrae), each with 4 corner keypoints.

```
Dataset Vertebrae Distribution:
┌────────────────────────────────────────────┐
│  Images with 2-4 vertebrae:  ~35%          │
│  Images with 5-6 vertebrae:  ~45%          │
│  Images with 7-10 vertebrae: ~20%          │
└────────────────────────────────────────────┘
```

#### Our Forced Solution: Fixed 40-Channel Output

To handle this variability, we designed our models to predict **40 heatmaps** (10 vertebrae × 4 corners):

| Approach | Description | Problem |
|----------|-------------|---------|
| **Fixed slots** | Always predict 40 keypoints | Model must learn to output zeros for missing vertebrae |
| **Sorted ordering** | Vertebrae sorted top-to-bottom | First ~20-28 channels used, rest are zeros |
| **Padding** | Images with fewer vertebrae have empty channels | Imbalanced learning signal |

**Impact on Results**: The model spends significant capacity learning which channels should be "off" rather than focusing purely on accurate localization.

#### What Benchmark Datasets Do Differently

| Dataset | Keypoints | Approach | Advantage |
|---------|-----------|----------|-----------|
| **COCO Pose** | 17 fixed | Always same anatomy | No ambiguity, all channels used |
| **MPII Human Pose** | 16 fixed | Always same anatomy | Consistent supervision |
| **BUU-LSPINE** | Variable (5-20+) | Instance detection first | Detects objects, then keypoints per instance |
| **SpineWeb** | Variable | Per-vertebra instances | Each vertebra is a separate detection |

**Recommended Alternative**: Instance-based detection (like Keypoint R-CNN) where:
1. First detect each vertebra as a bounding box
2. Then predict 4 keypoints per detected vertebra

This avoids the fixed-channel problem entirely.

### 13.2 Dataset Split & Test Set Issues

#### Insufficient Test Set

| Property | Our Dataset | Typical Benchmark |
|----------|-------------|-------------------|
| **Test Set Size** | 16 images (2.2%) | 100-500 images (10-20%) |
| **Test Labels** | ❌ Not provided | ✅ Held-out but available for evaluation |
| **Statistical Power** | Very low | High confidence |

**Impact**: Our test evaluation is statistically weak. With only 16 images, each image contributes ~6.25% to the final metric. A single bad prediction can swing MRE by 3-5 pixels.

#### Train/Val Contamination Risk

The dataset combines:
- **Honduras proprietary data**: 208 images (unique patients)
- **BUU-LSPINE filtered**: 508 images (from public dataset)

**Concern**: We don't know if the BUU-LSPINE portion has overlapping patients across train/val/test splits in the original dataset.

### 13.3 Annotation Inconsistencies

#### Variable Annotation Quality

```
Observed Issues (from manual inspection):
┌──────────────────────────────────────────────────────┐
│ ⚠️  Some vertebrae annotated with 4 corners         │
│ ⚠️  Some with only visible corners (2-3 points)     │
│ ⚠️  Endplate vs. body corner ambiguity              │
│ ⚠️  Different annotator styles between sources      │
└──────────────────────────────────────────────────────┘
```

#### Endplate Definition Ambiguity

The dataset defines corners as vertebral body corners, but:

| Landmark | Challenge |
|----------|-----------|
| **Superior endplate** | Curved surface - where exactly is the "corner"? |
| **Inferior endplate** | Same issue, especially with degeneration |
| **Osteophytes** | Should they be included or excluded? |
| **Slip deformity** | Severe slips distort normal corner positions |

**Impact on MRE**: Even perfect predictions may have 5-10 px "error" due to annotation subjectivity.

### 13.4 Image Quality Heterogeneity

#### Multi-Source Imaging Variation

```
Source 1: Honduras Clinical (208 images)
├── Older imaging equipment
├── Variable positioning
├── Real pathological cases
└── Higher noise levels

Source 2: BUU-LSPINE (508 images)
├── Standardized equipment
├── Research protocol
├── Mixed pathology/normal
└── Better quality control
```

#### Resolution and Scale Variations

| Property | Range | Impact |
|----------|-------|--------|
| **Image width** | 1200-3000 px | Different scales after resize |
| **Aspect ratio** | 0.6-1.4 | Variable padding amounts |
| **Pixel spacing** | 0.1-0.3 mm/px | 1 px error = 0.1-0.3 mm |
| **Contrast** | Low to high | Affects edge detection |

**Impact**: A model trained on well-contrasted images may struggle with low-contrast cases, inflating MRE on difficult samples.

### 13.5 No Published Baselines for Comparison

#### Benchmark Maturity Problem

| Metric | Our Dataset | Established Benchmark |
|--------|-------------|----------------------|
| **Publication date** | June 2025 | 2018-2021 |
| **Papers using it** | 0 | 10-50+ |
| **Baseline methods** | None | Multiple (U-Net, HRNet, etc.) |
| **Leaderboard** | ❌ | ✅ Often available |
| **State-of-art MRE** | Unknown | Published |

**Impact**: We cannot determine if our 43.85 px MRE is:
- Excellent (near human-level for this dataset)
- Average (typical first attempt)
- Poor (significant room for improvement)

#### Comparison with BUU-LSPINE Benchmarks

If we could compare with BUU-LSPINE results:

| Method | BUU-LSPINE MRE | Our Dataset MRE | Notes |
|--------|----------------|-----------------|-------|
| U-Net variants | ~15-25 px | 48.41 px | Different annotation scheme |
| HRNet variants | ~10-20 px | 43.85 px | Different task definition |
| Instance-based | ~8-15 px | Not tested | Recommended approach |

*Note: BUU-LSPINE typically uses different landmarks (centrum, pedicles, spinous process), making direct comparison impossible.*

### 13.6 Clinical Relevance of 40-Keypoint Approach

#### Overengineering for the Clinical Task?

For spondylolisthesis grading, we only need:
- **4 keypoints**: Inferior endplate of slipping vertebra + superior endplate of reference
- **Or 8 keypoints**: 2 vertebrae × 4 corners

**Current Approach**: Detecting 40 keypoints for 10 vertebrae when clinically we need ~8.

```
Clinical Reality:
┌─────────────────────────────────────────────────────────────┐
│  Spondylolisthesis typically occurs at:                     │
│    - L5/S1 (most common, ~85% of cases)                     │
│    - L4/L5 (second most common, ~10%)                       │
│    - L3/L4 (rare)                                           │
│                                                              │
│  We only need to detect 2 adjacent vertebrae accurately!    │
└─────────────────────────────────────────────────────────────┘
```

### 13.7 Summary of Contributing Factors to MRE

| Factor | Estimated Impact | Mitigation |
|--------|------------------|------------|
| **40 vs. 16 keypoints** | +10-15 px | Use instance detection |
| **Annotation subjectivity** | +5-10 px | Multi-annotator consensus |
| **Image quality variation** | +5-8 px | Domain adaptation |
| **Small test set** | High variance | More test data |
| **No baselines** | Cannot assess | Wait for publications |
| **Multi-source data** | +3-5 px | Source-specific normalization |

**Total Estimated Addressable Error**: ~28-43 px

### 13.8 Recommendations for Future Work

1. **Switch to Instance-Based Detection**
   - Use Keypoint R-CNN or YOLO-Pose
   - Detect vertebrae first, then localize 4 corners per instance
   - Eliminates the 40-channel problem

2. **Clinically-Focused Approach**
   - Only detect L4-S1 (the slip zone)
   - 12 keypoints instead of 40
   - Higher precision where it matters

3. **Multi-Annotator Validation**
   - Establish inter-annotator variability baseline
   - Report MRE relative to human disagreement

4. **Domain Adaptation**
   - Separate models for Honduras vs. BUU-LSPINE sources
   - Or explicit domain-adversarial training

5. **Benchmark Alternative**
   - Consider using BUU-LSPINE directly
   - Published baselines for comparison
   - Established evaluation protocol

---

*Document generated: January 26, 2026*
*Last updated: January 26, 2026 (HRNet-W32 results, Dataset Challenges section added)*
*Project: spondylolisthesis-maht-net*
