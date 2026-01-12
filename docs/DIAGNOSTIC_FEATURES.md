# Training Diagnostic Features

## Overview
Enhanced training diagnostics to help understand model behavior and identify issues during training.

## New Diagnostic Outputs

### 1. Real-Time Progress Bar Enhancements

**What you'll see:**
```
Epoch 1 [Train]: 16%|██| 40/247 [00:26<02:14, 1.54it/s, loss=886.84, MRE=226.6, SDR_2px=0.000, pred_range=[-1.2,6.5]]
```

**New fields:**
- **`pred_range=[min, max]`**: Shows the range of predicted heatmap values
  - Helps identify if model outputs are in reasonable ranges
  - Raw model outputs (no sigmoid) can be negative or very large
  - Should gradually converge to target range as training progresses

### 2. Heatmap Statistics (After Each Training Epoch)

**What you'll see:**
```
  Heatmap Statistics (Train):
    Predictions: min=-5.32, max=12.45, mean=2.18
    Targets:     min=0.00, max=1000.00, mean=15.67
```

**What this tells you:**
- **Predictions min/max/mean**: Range of model's raw output values
  - Should start random (e.g., -10 to +10)
  - Should gradually shift toward target range as training progresses
  
- **Targets min/max/mean**: Ground truth heatmap values
  - Fixed based on your heatmap generation (Gaussian with amplitude=1000)
  - Predictions should eventually match this distribution

**What to look for:**
- ✅ Prediction range gradually approaching target range
- ⚠️ Predictions stuck at very different scale than targets → learning rate or loss function issue
- ⚠️ Predictions exploding (e.g., >10000) → gradient explosion, reduce learning rate

### 3. Per-Sample Error Distribution (After Each Validation Epoch)

**What you'll see:**
```
  Per-Sample Error Distribution (Val):
    Best (min):      12.34 px
    Median:          89.56 px
    Worst (max):     456.78 px
    90th percentile: 234.56 px
```

**What this tells you:**
- **Best**: Model's performance on easiest samples
  - Shows the model's "best case" capability
  - Should decrease over time
  
- **Median**: Typical performance across validation set
  - More robust than mean (not affected by outliers)
  - Target: <10px for good performance
  
- **Worst**: Performance on hardest samples
  - Identifies problematic cases
  - High worst-case error might indicate:
    - Difficult anatomical variations
    - Poor quality images
    - Annotation errors
    
- **90th percentile**: Performance threshold covering 90% of samples
  - Useful for understanding overall robustness
  - Clinical applications often care about worst 10%

**What to look for:**
- ✅ All metrics decreasing over epochs
- ✅ Best error approaching <10px → model learning the task
- ⚠️ Large gap between best and worst → some samples much harder
- ⚠️ Median not improving after many epochs → potential plateau

## Interpreting Training Behavior

### Healthy Training Progression

**Epoch 1:**
```
pred_range=[-5.2, 8.3]
Heatmap Statistics: Predictions: min=-5.32, max=12.45, mean=2.18
Per-Sample Error: Best=150px, Median=320px, Worst=580px
```

**Epoch 10:**
```
pred_range=[-1.2, 15.6]
Heatmap Statistics: Predictions: min=-2.15, max=18.34, mean=8.45
Per-Sample Error: Best=45px, Median=105px, Worst=280px
```

**Epoch 50:**
```
pred_range=[0.3, 120.5]
Heatmap Statistics: Predictions: min=-0.12, max=124.56, mean=12.34
Per-Sample Error: Best=3px, Median=8px, Worst=45px
```

### Warning Signs

**1. Predictions not changing:**
```
Epoch 1:  pred_range=[-2.3, 4.5]
Epoch 10: pred_range=[-2.1, 4.8]  ⚠️ Barely moved!
Epoch 20: pred_range=[-2.0, 5.1]
```
→ **Action**: Increase learning rate or check loss function

**2. Predictions exploding:**
```
Epoch 5:  pred_range=[-1.2, 8.5]
Epoch 6:  pred_range=[-5.4, 45.3]
Epoch 7:  pred_range=[-15.2, 234.5]  ⚠️ Exploding!
```
→ **Action**: Reduce learning rate, add gradient clipping

**3. Error not improving:**
```
Epoch 10: Median=250px
Epoch 20: Median=245px
Epoch 30: Median=248px  ⚠️ Plateaued!
```
→ **Action**: Adjust learning rate, check architecture, verify data quality

**4. Huge variance in errors:**
```
Best=5px, Median=12px, Worst=450px  ⚠️ Huge gap!
```
→ **Action**: Inspect worst-case samples, check for annotation issues

## How to Use This Information

### During Training
1. **Monitor pred_range in progress bar**
   - Should gradually increase and shift toward target range
   - Sudden changes indicate instability

2. **Check heatmap statistics after each epoch**
   - Compare prediction range to target range
   - Look for gradual convergence

3. **Review per-sample distribution after validation**
   - Identify if model improving across all samples
   - Spot problematic outliers

### After Training
1. **Compare best vs worst samples**
   - Visualize predictions for samples at each percentile
   - Understand what makes certain cases difficult

2. **Track improvement rate**
   - Plot median error over epochs
   - Determine if more training needed or plateau reached

3. **Clinical validation**
   - If 90th percentile <20px → likely clinically useful
   - If worst-case >100px → investigate those specific cases

## Expected Target Performance

For vertebra keypoint detection (512×512 images):
- **Excellent**: Median MRE <5px, 90th percentile <10px
- **Good**: Median MRE <10px, 90th percentile <20px  
- **Acceptable**: Median MRE <15px, 90th percentile <30px
- **Poor**: Median MRE >20px

## Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| pred_range not moving | Learning rate too low | Increase LR by 2-5x |
| pred_range exploding | Learning rate too high | Decrease LR by 5-10x |
| Median error plateau | Model capacity insufficient | Increase model size or change architecture |
| Large best/worst gap | Outliers in dataset | Inspect and clean worst samples |
| Loss decreasing but MRE not | Loss-metric mismatch | Consider different loss function |
| Good training, poor val | Overfitting | Add regularization or more data |

## Files Modified

- **`training/unet_trainer.py`**: Added diagnostic tracking and printing
  - Heatmap statistics collection (every 50 batches)
  - Per-sample error tracking
  - Enhanced progress bar output
