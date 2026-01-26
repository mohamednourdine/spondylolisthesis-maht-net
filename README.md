# Spondylolisthesis MAHT-Net Project

**Goal**: Establish the first baseline performance metrics for automated spondylolisthesis grading using deep learning.

This project implements and benchmarks four models: **U-Net**, **ResNet Keypoint Detector**, **Keypoint R-CNN**, and **MAHT-Net** on the Spondylolisthesis Vertebral Landmark Dataset.

---

## 🎯 Current Results

### Best Model: HRNet-W32 🏆

| Metric | HRNet-W32 | ResNet-50 | UNet |
|--------|-----------|-----------|------|
| **Val MRE** | **43.85 px** | 51.06 px | 48.41 px |
| **Val SDR@24px** | **43.65%** | 36.1% | 38.0% |
| Val SDR@18px | 42.28% | - | 36.4% |
| Val SDR@12px | 38.82% | - | 31.9% |
| Val SDR@6px | 26.14% | - | 19.5% |
| Parameters | 31.78M | 30.79M | 8.65M |

*Trained on 494 images, validated on 204 images, tested on 16 images*

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
| **UNet Model** | ✅ Complete | Architecture, training, evaluation, **baseline established** |
| **Training System** | ✅ Complete | Multi-model framework with metrics tracking |
| **Metrics & Evaluation** | ✅ Complete | MRE, MSE, SDR (6px, 12px, 18px, 24px) |
| **Visualization Tools** | ✅ Complete | Prediction overlay, heatmap comparison |
| **Test Evaluation** | ✅ Complete | Inference on held-out test set |
| **Project Organization** | ✅ Complete | Clean structure, tests, experiment management |
| **MAHT-Net** | 📋 Planned | Implementation pending |
| **ResNet-Keypoint** | 📋 Planned | Implementation pending |
| **Keypoint-RCNN** | 📋 Planned | Implementation pending |

**Current Focus**: UNet baseline complete. Ready for MAHT-Net implementation.

### 🎯 Implementation Status

- ✅ **Complete Training Infrastructure**
  - Global KeypointEvaluator for consistent metrics (MRE, MSE, SDR)
  - BaseTrainer for code reuse across models
  - Model Registry for centralized model management
  - Automatic experiment organization by model type
  - Mac/MPS optimization for Apple Silicon
  
- ✅ **UNet Implementation** (Baseline Complete)
  - 17.27M parameters, encoder-decoder architecture
  - Heatmap-based keypoint detection (40 channels for 10 vertebrae × 4 corners)
  - Weighted MSE loss with keypoint emphasis
  - Dropout regularization (30% at bottleneck)
  - Enhanced augmentation (rotation, brightness, contrast, noise)
  - Val MRE: 66-71 px, Val SDR@24px: 28.2%

- 📋 **Pending Models**
  - ResNet-Keypoint: Direct coordinate regression
  - Keypoint R-CNN: Instance-aware detection
  - MAHT-Net: Multi-scale attention heatmap transformer

## 📈 Metrics Tracked

All models are evaluated with comprehensive metrics during training:

- **MRE (Mean Radial Error)**: Average Euclidean distance between predicted and ground truth keypoints (lower is better)
- **MSE (Mean Squared Error)**: Squared distance metric, emphasizes large errors (lower is better)
- **SDR (Successful Detection Rate)**: Percentage of keypoints detected within threshold (higher is better)
  - SDR@6px, SDR@12px, SDR@18px, SDR@24px (pixel-based thresholds)

> **Note**: Using pixel-based metrics as the dataset (JPG format) lacks calibration metadata for mm conversion.

### UNet Baseline Results (Achieved)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **MRE** | ~45 px | 48.41 px | N/A* |
| **MSE** | ~4,500 px² | ~4,500 px² | N/A* |
| **SDR@6px** | ~22% | 19.5% | N/A* |
| **SDR@12px** | ~28% | 31.9% | N/A* |
| **SDR@18px** | ~32% | 36.4% | N/A* |
| **SDR@24px** | ~35% | 38.0% | N/A* |

*Test set has no ground truth labels (blind evaluation)*

### Model Performance Summary

| Model | Val MRE (px) | Val SDR@24px (%) | Status |
|-------|--------------|------------------|--------|
| U-Net | 48.41 | 38.0% | ✅ **Trained** |
| ResNet-50 | 51.06 | 36.1% | ✅ **Trained** |
| **HRNet-W32** | **43.85** | **43.65%** | ✅ **Best** 🏆 |
| Keypoint R-CNN | < 40 | > 45 | 📋 Planned |
| MAHT-Net | < 30 | > 55 | 📋 Planned |

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data understanding & exploration |
| Phase 2 | ✅ Complete | UNet baseline implementation & training |
| Phase 3 | ✅ Complete | ResNet-50 & HRNet-W32 training |
| Phase 4 | ⏳ In Progress | MAHT-Net & final evaluation |
| Phase 5 | ⏳ Pending | Paper writing & submission |

**Current Focus**: HRNet-W32 achieves best results (MRE 43.85 px, SDR@24px 43.65%). Proceeding with MAHT-Net.

---

## 🛠️ Scripts Reference

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_mac.py` | Mac-optimized training with MPS acceleration |
| `train.py` | Full training for cloud/GPU environments |
| `scripts/train_unet.py` | Standalone UNet training script |

### Evaluation & Visualization Scripts

| Script | Description |
|--------|-------------|
| `scripts/visualize_predictions.py` | Visualize predictions vs ground truth |
| `scripts/evaluate_test.py` | Run inference on test set |
| `scripts/compare_experiments.py` | Compare training experiments |
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

## 📈 Expected Results

### Achieved Results (UNet Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| **Val MRE** | 66-71 px | ~17% of image width |
| **Val SDR@24px** | 28.2% | 28% within 24 pixels |
| **Training Time** | 2.6 hours | 50 epochs on M1 Mac |
| **Model Size** | 17.27M params | Base channels: 64 |

### Model Architecture Details

```
UNet Configuration:
├── Input: 512 x 512 x 3 (RGB)
├── Encoder: 5 levels (64 → 128 → 256 → 512 → 512)
├── Decoder: 5 levels with skip connections
├── Output: 512 x 512 x 40 (10 vertebrae × 4 corners)
├── Dropout: 30% at bottleneck
├── Heatmap Sigma: 15 pixels
└── Loss: Weighted MSE (keypoint weight: 5x, background: 0.05x)
```

### Target Performance (Other Models)

| Model | Target MRE | Target SDR@24px | Priority |
|-------|------------|-----------------|----------|
| ResNet-Keypoint | < 50 px | > 35% | High |
| Keypoint R-CNN | < 40 px | > 45% | Medium |
| MAHT-Net | < 30 px | > 55% | High |

*Note: These are relative improvements expected over UNet baseline.*

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

# Train on Mac (2.5 hours)
python train_mac.py --epochs 50 --batch-size 4

# Train full model in cloud
python train.py --model unet --epochs 50 --batch-size 8 --experiment-name production_v1

# Visualize predictions
python scripts/visualize_predictions.py --num-samples 10 --heatmaps

# Evaluate on test set
python scripts/evaluate_test.py

# Compare results
python scripts/compare_experiments.py --model unet
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