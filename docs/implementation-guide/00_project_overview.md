# MAHT-Net Implementation Guide

## Project Overview

**Goal**: Build a state-of-the-art vertebral corner point detection system for spondylolisthesis assessment using the BUU-LSPINE dataset.

**Target**: Beat Klinwichit et al. baseline (MED 4.63mm AP, 4.91mm LA) with target MED < 4.0mm (>15% improvement).

---

## Training Environment

| Component | Specification |
|-----------|---------------|
| Machine | MacBook Pro (Mac15,6) |
| Chip | **Apple M3 Pro** |
| CPU | 12-core (6P + 6E) |
| GPU | 18-core Apple GPU |
| Memory | 18 GB unified |
| Backend | **MPS (Metal Performance Shaders)** |

**Note**: Training will use PyTorch with MPS backend for Apple Silicon acceleration.

---

## Current Project Status (February 2025)

### ✅ COMPLETED

| Component | Location | Status |
|-----------|----------|--------|
| Architecture Design | `docs/MAHT_NET_ARCHITECTURE.md` | Complete (1160 lines) |
| Component Documentation | `docs/architecture/` | 6 detailed files |
| ResNet-50 Heatmap Model | `models/resnet_heatmap.py` | Fully implemented (~322 lines) |
| U-Net Model | `models/unet.py` | Implemented |
| Loss Functions | `training/losses.py` | CornerNet focal loss |
| Training System | `train.py` | Working entry point |
| Configuration | `config/base_config.py` | Base config ready |

### ⚠️ PARTIALLY IMPLEMENTED

| Component | Issue | Action Required |
|-----------|-------|-----------------|
| Dataset Loader | Expects JSON, but BUU-LSPINE uses CSV | Adapt `src/data/dataset.py` |
| Config Paths | Points to `data/Train/` | Update to `data/buu-lspine/` |

### ❌ NOT IMPLEMENTED (STUBS)

| Component | Location | Issue |
|-----------|----------|-------|
| **MAHT-Net Model** | `models/maht_net.py` | Only 35 lines - basic CNN stub |
| HRNet Model | `models/hrnet_heatmap.py` | Need to verify |

---

## BUU-LSPINE Dataset Summary

### Dataset Statistics

```
Total Patients: 3,600
Total Images: 7,200

├── AP View (Anterior-Posterior)
│   ├── Images: 3,600
│   ├── Keypoints per image: 20 (10 vertebrae × 2 corners)
│   └── Location: data/buu-lspine/AP/
│
└── LA View (Lateral)
    ├── Images: 3,600
    ├── Keypoints per image: 22 (11 edges × 2 corners)
    └── Location: data/buu-lspine/LA/
```

### File Naming Convention

```
{PatientID}-{Gender}-{Age}Y{View}.{extension}

Examples:
  3065-F-046Y0.jpg  →  Patient 3065, Female, 46 years, AP view (Y0)
  3065-F-046Y0.csv  →  Corresponding annotations
  3065-F-046Y1.jpg  →  Same patient, LA view (Y1)
```

### Annotation Format (CSV)

Each CSV file contains corner coordinates:

```csv
x1,y1,x2,y2,label
876.22,167.06,1111.47,168.77,0    # Row 1: Top edge of L1
865.99,313.67,1119.99,313.67,0    # Row 2: Bottom edge of L1
...
```

| Column | Description |
|--------|-------------|
| x1, y1 | Left corner coordinate |
| x2, y2 | Right corner coordinate |
| label | Spondylolisthesis status (0=normal, 1=affected) |

**AP View**: 10 rows (10 vertebral edges: L1-top to L5-bottom)
**LA View**: 11 rows (includes S1 reference)

---

## Baseline to Beat

### Klinwichit et al. 2023 Results

| View | Model | MED (mm) | Classification |
|------|-------|----------|----------------|
| AP | ResNet152V2 | **4.63** | 95.14% (SVM) |
| LA | ResNet152V2 | **4.91** | 92.26% (SVM) |

### Our Targets

| Metric | Target | Improvement |
|--------|--------|-------------|
| MED AP | < 4.0 mm | > 13% |
| MED LA | < 4.0 mm | > 18% |
| Classification | > 95% | End-to-end |

---

## Implementation Roadmap

### Phase 1: Dataset Infrastructure (1-2 days)
**Document**: [01_dataset_preparation.md](01_dataset_preparation.md)

1. Create BUU-LSPINE specific DataLoader
2. Implement CSV annotation parser
3. Setup train/val/test splits (70/15/15)
4. Add data augmentation pipeline
5. Verify pixel spacing calibration

### Phase 2: MAHT-Net Implementation (5-7 days)
**Document**: [02_maht_net_implementation.md](02_maht_net_implementation.md)

1. CNN Backbone (EfficientNetV2-S) - 1 day
2. Transformer Bridge - 1 day
3. Vertebral Attention Module (VAM) - 2 days
4. Multi-scale Decoder - 1 day
5. DARK Decoding - 1 day
6. Integration & Testing - 1 day

### Phase 3: Training & Experiments (3-5 days)
**Document**: [03_training_pipeline.md](03_training_pipeline.md)

1. Train MAHT-Net on BUU-LSPINE (Mac M3 Pro with MPS)
2. Compare against Klinwichit et al. baseline (4.63mm AP, 4.91mm LA)
3. Run ablation studies
4. Document results

### Phase 4: Evaluation & Paper (2-3 days)
**Document**: [04_evaluation_protocol.md](04_evaluation_protocol.md)

1. Full test set evaluation
2. Statistical analysis
3. Generate visualizations
4. Write paper results section

---

## Quick Start Commands

```bash
# Activate environment
conda activate phd
cd /Users/mnourdine/phd/spondylolisthesis-maht-net

# Verify dataset
ls data/buu-lspine/AP/*.jpg | wc -l  # Should be 3600
ls data/buu-lspine/LA/*.csv | wc -l  # Should be 3600

# Test existing ResNet model
python -c "from models.resnet_heatmap import ResNetHeatmap; m = ResNetHeatmap(num_keypoints=20); print('ResNet OK')"

# Test MAHT-Net (currently stub)
python -c "from models.maht_net import MAHTNet; m = MAHTNet(); print('MAHT-Net loaded (stub)')"
```

---

## Documentation Map

```
docs/implementation-guide/
├── 00_project_overview.md        ← YOU ARE HERE
├── 01_dataset_preparation.md     ← Dataset & DataLoader setup
├── 02_maht_net_implementation.md ← Step-by-step model implementation
├── 03_training_pipeline.md       ← Training on Colab
├── 04_evaluation_protocol.md     ← Metrics & evaluation
└── 05_experiments_checklist.md   ← Ablation studies & experiments

docs/architecture/
├── 01_cnn_backbone.md            ← EfficientNetV2-S details
├── 02_transformer_bridge.md      ← 4-layer transformer
├── 03_vertebral_attention_module.md ← VAM (key contribution)
├── 04_multiscale_decoder.md      ← Skip connections
├── 05_dark_decoding.md           ← Sub-pixel accuracy
└── 06_loss_functions.md          ← Combined loss
```

---

## Next Steps

1. **Start with**: [01_dataset_preparation.md](01_dataset_preparation.md)
2. **Priority**: Fix dataset loader to read CSV annotations
3. **Goal**: Get training pipeline working with BUU-LSPINE data

---

*Last Updated: February 2025*
