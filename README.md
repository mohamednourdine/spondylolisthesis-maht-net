# Spondylolisthesis MAHT-Net Project

**Goal**: Establish the first baseline performance metrics for automated spondylolisthesis grading using deep learning.

This project implements and benchmarks deep learning models for vertebral landmark detection: **U-Net**, **ResNet-50**, **HRNet-W32**, and **MAHT-Net** (planned) on the Spondylolisthesis Vertebral Landmark Dataset.

---

## 🎯 Current Results

### Best Model: HRNet-W32 🏆

| Metric | HRNet-W32 | UNet | ResNet-50 |
|--------|-----------|------|-----------|
| **Val MRE** | **43.85 px** | 48.41 px | 51.06 px |
| **Val SDR@24px** | **43.65%** | 38.0% | 36.1% |
| Val SDR@18px | 42.28% | 36.4% | - |
| Val SDR@12px | 38.82% | 31.9% | - |
| Val SDR@6px | 26.14% | 19.5% | - |
| Parameters | 31.78M | 8.65M | 30.79M |
| Training Time | 4.71 hrs | ~4 hrs | 7.90 hrs |
| Best Epoch | 87/100 | 82/100 | 86/100 |

*Trained on 496 images, validated on 204 images, tested on 16 images (no ground truth)*

### Key Achievements
- ✅ **HRNet-W32**: 43.85 px MRE (32.6% better than UNet baseline)
- ✅ **ImageNet pretrained backbone** with frozen layers + differential learning rates
- ✅ **Cosine annealing scheduler** with 5-epoch warmup
- ✅ **Per-layer dropout** and MC Dropout for uncertainty estimation

---

## 🚀 Quick Start

**New to the project?** Follow these guides in order:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands and workflows
2. **[QUICK_START.md](docs/QUICK_START.md)** - Get training started TODAY (2-3 hours)
3. **[training_guide.md](docs/training_guide.md)** - Complete training documentation
4. **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Solutions to common issues

### 🔥 Test Locally First (Recommended)

Before cloud training, verify everything works with a small dataset:

```bash
conda activate phd
python tests/test_training_small.py
```

This runs 2 epochs on 10 samples (~2 minutes) to verify the complete training pipeline.

### 🍎 Mac Training (Apple Silicon)

Optimized training for Mac with MPS acceleration:

```bash
# Full training (50 epochs, ~2.5 hours on M1/M2)
python train_mac.py --epochs 50 --batch-size 4

# Quick test (5 epochs)
python train_mac.py --epochs 5 --batch-size 4
```

---

## Installation

### Environment Setup

```bash
# Clone repository
git clone git@github.com:mohamednourdine/spondylolisthesis-maht-net.git
cd spondylolisthesis-maht-net

# Create conda environment
conda create -n phd python=3.9 -y
conda activate phd

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Dataset Download

Download the Spondylolisthesis Vertebral Landmark Dataset:

```bash
# Dataset URL
https://data.mendeley.com/datasets/5jdfdgp762/1
```

**Dataset Statistics:**
- **Total Images**: 716 (208 Honduran + 508 BUU-LSPINE)
- **Train**: 494 images with JSON annotations
- **Validation**: 204 images with JSON annotations  
- **Test**: 16 images (blind evaluation, no labels)
- **Vertebrae per image**: 5-10 (avg ~7)
- **Keypoints per vertebra**: 4 corners (TL, TR, BL, BR)
- **Image format**: JPG (no DICOM calibration metadata)

```
# Expected structure after extraction:
data/
├── Train/
│   └── Keypointrcnn_data/
│       ├── images/
│       │   ├── train/  (494 images)
│       │   └── val/    (204 images)
│       └── labels/
│           ├── train/  (494 JSON files)
│           └── val/    (204 JSON files)
└── Test/               (16 images, no labels)
```

---

## Usage

### 1. Data Exploration

Explore and understand the dataset:

```bash
# Activate environment
conda activate phd

# Run data understanding notebook
jupyter notebook notebooks/00_data_understanding.ipynb

# Run preprocessing pipeline
jupyter notebook notebooks/01_preprocessing_pipeline.ipynb
```

### 2. Model Training

#### Quick Test (Local PC - Small Dataset)
```bash
# Test with 10 samples (2 minutes)
python tests/test_training_small.py
```

#### Full Training (Cloud - Complete Dataset)
```bash

cd drive/Othercomputers/My Mac/phd/spondylolisthesis-maht-net
# Train U-Net (50 epochs, ~6-8 hours on GPU)
# Batch size 8 is optimal for T4 GPU (15GB) - use 4 if OOM errors occur
python train.py --model unet --epochs 50 --batch-size 8 --experiment-name production_v1

# Train MAHT-Net (when implemented)
python train.py --model maht-net --epochs 50 --batch-size 8 --experiment-name production_v1

# Train ResNet-Keypoint (when implemented)
python train.py --model resnet-keypoint --epochs 50 --batch-size 8 --experiment-name production_v1

# Train Keypoint-RCNN (when implemented)
python train.py --model keypoint-rcnn --epochs 50 --batch-size 8 --experiment-name production_v1
```

#### Resume Training
```bash
python train.py --model unet \
  --resume experiments/results/unet/production_v1_20251216_104526/checkpoints/last_model.pth
```

See **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** and **[training_guide.md](docs/training_guide.md)** for detailed instructions.

### 3. Visualization & Analysis

Visualize model predictions on validation data:

```bash
# Visualize 10 random validation samples with heatmaps
python scripts/visualize_predictions.py --num-samples 10 --heatmaps

# Visualize specific number of samples
python scripts/visualize_predictions.py --num-samples 20

# Use training split instead
python scripts/visualize_predictions.py --split train --num-samples 5
```

Output saved to: `experiments/visualizations/`

### 4. Test Set Evaluation

Run inference on held-out test images:

```bash
# Evaluate on test set (16 images)
python scripts/evaluate_test.py

# Use custom checkpoint
python scripts/evaluate_test.py --checkpoint path/to/best_model.pth
```

Output saved to: `experiments/test_evaluation/`
- `test_predictions.json` - Keypoint coordinates
- `test_summary.csv` - Detection summary
- `visualizations/` - Prediction images

### 5. Evaluation & Comparison

Compare training results and find best models:

```bash
# Compare all UNet experiments
python scripts/compare_experiments.py --model unet

# Show top 5 best experiments across all models
python scripts/compare_experiments.py --list-all --top 5

# Show detailed metrics for specific experiment
python scripts/compare_experiments.py --model unet --details "production_v1"
```

Results are saved in `experiments/results/{model}/` with checkpoints, configs, and training history.

---

## 📚 Documentation

Comprehensive guides available:

### Quick References
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands and workflows ⭐
- **[QUICK_START.md](docs/QUICK_START.md)** - Get started in 2-3 hours
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Project Organization
- **[PROJECT_ORGANIZATION.md](docs/PROJECT_ORGANIZATION.md)** - Project structure and organization
- **[tests/README.md](tests/README.md)** - Test scripts documentation
- **[experiments/results/README.md](experiments/results/README.md)** - Results structure

## 📊 Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Data Pipeline** | ✅ Complete | Dataset loading, preprocessing, augmentation |
| **UNet Model** | ✅ Complete | Baseline model, Val MRE: 48.41 px |
| **ResNet-50 Model** | ✅ Complete | Heatmap regression, Val MRE: 51.06 px |
| **HRNet-W32 Model** | ✅ Complete | **Best model**, Val MRE: 43.85 px 🏆 |
| **Training System** | ✅ Complete | Multi-model framework with metrics tracking |
| **Metrics & Evaluation** | ✅ Complete | MRE, MSE, SDR (6px, 12px, 18px, 24px) |
| **Visualization Tools** | ✅ Complete | Prediction overlay, heatmap comparison, training curves |
| **Test Evaluation** | ✅ Complete | Inference on held-out test set |
| **Project Organization** | ✅ Complete | Clean structure, tests, experiment management |
| **MAHT-Net** | 📋 Planned | Multi-scale Attention Hybrid Transformer |
| **Keypoint-RCNN** | 📋 Planned | Instance-based detection approach |

**Current Focus**: HRNet-W32 achieves best results. Proceeding with MAHT-Net implementation.

### 🎯 Implementation Status

- ✅ **Complete Training Infrastructure**
  - Global KeypointEvaluator for consistent metrics (MRE, MSE, SDR)
  - BaseTrainer for code reuse across models
  - Model Registry for centralized model management
  - Automatic experiment organization by model type
  - Mac/MPS optimization for Apple Silicon
  
- ✅ **UNet Implementation** (Baseline)
  - 8.65M parameters, encoder-decoder architecture
  - Heatmap-based keypoint detection (40 channels for 10 vertebrae × 4 corners)
  - Weighted MSE loss with keypoint emphasis
  - Val MRE: 48.41 px, Val SDR@24px: 38.0%

- ✅ **ResNet-50 Implementation**
  - 30.79M parameters, pretrained backbone
  - Heatmap regression with bilinear upsampling
  - Val MRE: 51.06 px, Val SDR@24px: 36.1%

- ✅ **HRNet-W32 Implementation** (Best Model)
  - 31.78M parameters, ImageNet pretrained via `timm`
  - Parallel multi-resolution branches with cross-scale fusion
  - Differential learning rates (backbone: 1e-5, head: 1e-4)
  - Cosine annealing with 5-epoch warmup
  - Val MRE: 43.85 px, Val SDR@24px: 43.65%

- 📋 **Pending Models**
  - Keypoint R-CNN: Instance-aware detection
  - MAHT-Net: Multi-scale attention heatmap transformer

## 📈 Metrics Tracked

All models are evaluated with comprehensive metrics during training:

- **MRE (Mean Radial Error)**: Average Euclidean distance between predicted and ground truth keypoints (lower is better)
- **MSE (Mean Squared Error)**: Squared distance metric, emphasizes large errors (lower is better)
- **SDR (Successful Detection Rate)**: Percentage of keypoints detected within threshold (higher is better)
  - SDR@6px, SDR@12px, SDR@18px, SDR@24px (pixel-based thresholds)

> **Note**: Using pixel-based metrics as the dataset (JPG format) lacks calibration metadata for mm conversion.

### Model Performance Summary (All Trained Models)

| Model | Val MRE (px) | Val SDR@24px | Parameters | Training Time | Status |
|-------|--------------|--------------|------------|---------------|--------|
| **HRNet-W32** | **43.85** | **43.65%** | 31.78M | 4.71 hrs | 🏆 **Best** |
| UNet | 48.41 | 38.0% | 8.65M | ~4 hrs | ✅ Trained |
| ResNet-50 | 51.06 | 36.1% | 30.79M | 7.90 hrs | ✅ Trained |
| Keypoint R-CNN | < 40 | > 45% | - | - | 📋 Planned |
| MAHT-Net | < 35 | > 50% | - | - | 📋 Planned |

*Test set (16 images) has no ground truth labels - blind evaluation only*

### Improvement Over Baseline

| Comparison | MRE Improvement | SDR@24px Improvement |
|------------|-----------------|----------------------|
| HRNet vs UNet | -9.4% (4.56 px) | +14.9% (5.65 pts) |
| HRNet vs ResNet | -14.1% (7.21 px) | +20.9% (7.55 pts) |

### Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data understanding & exploration |
| Phase 2 | ✅ Complete | UNet baseline implementation & training |
| Phase 3 | ✅ Complete | ResNet-50 & HRNet-W32 training |
| Phase 4 | ⏳ In Progress | MAHT-Net & Keypoint R-CNN implementation |
| Phase 5 | ⏳ Pending | Paper writing & submission |

**Current Focus**: HRNet-W32 achieves best results (MRE 43.85 px, SDR@24px 43.65%). Proceeding with MAHT-Net.

### Dataset Challenges

See [docs/ADVISOR_MEETING_SUMMARY.md](docs/ADVISOR_MEETING_SUMMARY.md#13-dataset-challenges--their-impact-on-results) for detailed analysis of factors affecting results:
- Variable vertebrae count requiring 40-keypoint prediction
- Insufficient test set (16 images, no labels)
- Multi-source image quality variation
- No published baselines for comparison

---

## 🛠️ Scripts Reference

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_hrnet.py` | HRNet-W32 training with pretrained backbone |
| `train_resnet.py` | ResNet-50 heatmap regression training |
| `train_mac.py` | UNet training optimized for Mac/MPS |
| `train.py` | Full training for cloud/GPU environments |

### Evaluation & Visualization Scripts

| Script | Description |
|--------|-------------|
| `scripts/visualize_predictions.py` | Visualize predictions vs ground truth |
| `scripts/evaluate_test.py` | Run inference on test set (supports unet, resnet, hrnet) |
| `scripts/compare_experiments.py` | Compare training experiments |
| `scripts/generate_training_curves.py` | Generate training curves for all models |
| `scripts/diagnose_predictions.py` | Debug model predictions |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/calculate_calibration.py` | Analyze vertebra dimensions |
| `scripts/generate_augmented_data.py` | Generate augmented training data |
| `scripts/cleanup_experiments.py` | Clean up old experiment results |

### Test Scripts

| Script | Description |
|--------|-------------|
| `tests/test_training_small.py` | Quick 2-epoch training test |
| `tests/test_hrnet_quick.py` | HRNet model verification test |
| `tests/test_components.py` | Test individual components |
| `tests/test_unet.py` | UNet model tests |
| `tests/test_loss_functions.py` | Loss function tests |
| `tests/test_coordinate_scaling.py` | Coordinate system tests |

---

## 🎯 Research Goals

1. **Establish Baselines**: Create the first comprehensive baseline results on this dataset
2. **Model Comparison**: Compare 4 different architectures for vertebral landmark detection
3. **Clinical Validation**: Achieve clinically-relevant accuracy (MRE < 3mm, Grade accuracy > 90%)
4. **Open Source**: Release code and pretrained models for research community

---

## 📈 Model Architectures

### Best Model: HRNet-W32 Architecture

```
HRNet-W32 Configuration:
├── Input: 512 × 512 × 3 (RGB)
├── Backbone: HRNet-W32 (ImageNet pretrained via timm)
│   ├── Stage 1: 64 channels (1/4 resolution)
│   ├── Stage 2: 32, 64 channels (parallel branches)
│   ├── Stage 3: 32, 64, 128 channels
│   └── Stage 4: 32, 64, 128, 256 channels
├── Cross-scale Fusion: Upsampling + 1×1 conv
├── Head: 480 → 64 → 40 channels
├── Output: 512 × 512 × 40 (10 vertebrae × 4 corners)
└── Training: AdamW, cosine annealing, warmup 5 epochs
```

### All Model Architectures

| Model | Architecture | Backbone | Parameters |
|-------|--------------|----------|------------|
| **HRNet-W32** | Multi-resolution parallel branches | ImageNet pretrained | 31.78M |
| UNet | Encoder-decoder with skip connections | From scratch | 8.65M |
| ResNet-50 | Feature pyramid + upsampling | ImageNet pretrained | 30.79M |

### Target Performance (Pending Models)

| Model | Target MRE | Target SDR@24px | Notes |
|-------|------------|-----------------|-------|
| Keypoint R-CNN | < 40 px | > 45% | Instance-based detection |
| MAHT-Net | < 35 px | > 50% | Multi-scale attention transformer |

---

## 🤝 Contributing

This project establishes the first baselines on the Spondylolisthesis dataset. Contributions welcome:

- Improved model architectures
- Additional evaluation metrics
- Clinical validation
- Documentation improvements

---

## 📝 Citation

If you use this code or dataset, please cite:

```bibtex
@misc{spondylolisthesis_baselines_2025,
  title={Establishing Baseline Performance for Automated Spondylolisthesis Grading},
  author={Mogham Njikam Mohamed Nourdine},
  year={2025},
  note={First comprehensive benchmark on Spondylolisthesis Vertebral Landmark Dataset}
}
```

Dataset citation:
```bibtex
@data{mendeley_spondylo_2025,
  author={Reyes, et al.},
  title={Spondylolisthesis Vertebral Landmark Dataset},
  year={2025},
  publisher={Mendeley Data},
  version={V1},
  doi={10.17632/5jdfdgp762.1}
}
```

---

## 🐛 Troubleshooting

Having issues? Check:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands and workflows
2. **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Solutions to common problems
3. **Environment**: `conda activate phd` and verify PyTorch installation
4. **Quick Test**: Run `python tests/test_training_small.py` to verify setup
5. **Logs**: Check `experiments/results/{model}/` for training logs and metrics

---

## 📧 Contact

**Project Lead**: Mohamed Nourdine  
**Email**: mohamednjikam25@hotmail.com  
**GitHub**: [@mohamednourdine](https://github.com/mohamednourdine)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Spondylolisthesis Vertebral Landmark Dataset (Mendeley, 2025)
- **Inspiration**: MAHT-Net architecture for medical imaging
- **Community**: PyTorch and medical imaging research communities

---

**Ready to get started? → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) 🚀**

### Quick Commands

```bash
# Test locally first (2 minutes)
python tests/test_training_small.py

# Train HRNet-W32 (best model, ~5 hours on GPU)
python train_hrnet.py --epochs 100 --batch-size 8

# Train ResNet-50
python train_resnet.py --epochs 100 --batch-size 8

# Train UNet on Mac (2.5 hours)
python train_mac.py --epochs 50 --batch-size 4

# Visualize predictions
python scripts/visualize_predictions.py --num-samples 10 --heatmaps

# Evaluate on test set (supports unet, resnet, hrnet)
python scripts/evaluate_test.py --model-type hrnet

# Generate training curves comparison
python scripts/generate_training_curves.py

# Compare results
python scripts/compare_experiments.py --model hrnet
```

---

## 📂 Project Structure

```
spondylolisthesis-maht-net/
├── config/                     # Configuration files
│   ├── mac_config.py           # Mac training config
│   └── base_config.py          # Base configuration
├── data/                       # Dataset
│   ├── Train/                  # Training data (494 images)
│   ├── Validation/             # Validation data (204 images)
│   └── Test/                   # Test data (16 images, no labels)
├── evaluation/                 # Evaluation metrics
│   ├── keypoint_evaluator.py
│   ├── metrics.py
│   └── unet_metrics.py
├── experiments/                # Training outputs
│   ├── results/                # Model checkpoints & logs
│   ├── visualizations/         # Prediction visualizations
│   └── test_evaluation/        # Test set results
├── models/                     # Model architectures
│   ├── unet.py                 # UNet implementation
│   ├── resnet_keypoint.py      # ResNet-50 heatmap model
│   ├── hrnet_heatmap.py        # HRNet-W32 (best model)
│   └── maht_net.py             # MAHT-Net (planned)
├── scripts/                    # Utility scripts
├── src/                        # Core source code
│   ├── data/                   # Data loading & augmentation
│   └── utils/                  # Helper functions
├── tests/                      # Test scripts
├── training/                   # Training infrastructure
│   ├── base_trainer.py         # Base trainer class
│   ├── unet_trainer.py         # UNet trainer
│   └── losses.py               # Loss functions
├── train_mac.py                # Mac training entry point
└── train.py                    # Cloud training entry point
```