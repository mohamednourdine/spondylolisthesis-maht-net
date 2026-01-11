# Spondylolisthesis MAHT-Net Project

**Goal**: Establish the first baseline performance metrics for automated spondylolisthesis grading using deep learning.

This project implements and benchmarks four models: **U-Net**, **ResNet Keypoint Detector**, **Keypoint R-CNN**, and **MAHT-Net** on the Spondylolisthesis Vertebral Landmark Dataset.

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

# Extract to project directory
# Expected structure:
# data/
# ├── Train/
# │   ├── images/ (494 images)
# │   └── labels/ (494 JSON files)
# ├── Validation/
# │   ├── images/ (206 images)
# │   └── labels/ (206 JSON files)
# └── Clinical/
#     ├── images/ (16 images)
#     └── labels/ (16 JSON files)
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
# Train U-Net (50 epochs, ~6-8 hours on GPU)
python train.py --model unet --epochs 50 --batch-size 16 --experiment-name production_v1

# Train MAHT-Net (when implemented)
python train.py --model maht-net --epochs 50 --batch-size 16 --experiment-name production_v1

# Train ResNet-Keypoint (when implemented)
python train.py --model resnet-keypoint --epochs 50 --batch-size 16 --experiment-name production_v1

# Train Keypoint-RCNN (when implemented)
python train.py --model keypoint-rcnn --epochs 50 --batch-size 16 --experiment-name production_v1
```

#### Resume Training
```bash
python train.py --model unet \
  --resume experiments/results/unet/production_v1_20251216_104526/checkpoints/last_model.pth
```

See **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** and **[training_guide.md](docs/training_guide.md)** for detailed instructions.

### 3. Evaluation & Comparison

Compare training results and find best models:

```bash
# Compare all UNet experiments
python scripts/compare_experiments.py --model unet

# Show top 5 best experiments across all models
python scripts/compare_experiments.py --list-all --top 5

# Show detailed metrics for specific experiment
python scripts/compare_experiments.py --model unet --details "production_v1"
```

All results are automatically saved in `experiments/results/{model}/` with:
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
| **UNet Model** | ✅ Complete | Architecture, training, evaluation |
| **Training System** | ✅ Complete | Multi-model framework with metrics tracking |
| **Metrics & Evaluation** | ✅ Complete | MRE, SDR (2mm, 2.5mm, 3mm, 4mm) |
| **Project Organization** | ✅ Complete | Clean structure, tests, experiment management |
| **MAHT-Net** | 📋 Planned | Implementation pending |
| **ResNet-Keypoint** | 📋 Planned | Implementation pending |
| **Keypoint-RCNN** | 📋 Planned | Implementation pending |

**Current Focus**: Ready for full UNet training in cloud. Other models to follow same pattern.

### 🎯 Implementation Status

- ✅ **Complete Training Infrastructure**
  - Global KeypointEvaluator for consistent metrics
  - BaseTrainer for code reuse across models
  - Model Registry for centralized model management
  - Automatic experiment organization by model type
  
- ✅ **UNet Implementation**
  - 31M parameters, encoder-decoder architecture
  - Heatmap-based keypoint detection
  - Focal loss for better performance
  - Tested and working with small dataset

## 📈 Metrics Tracked

All models are evaluated with comprehensive metrics during training:

- **MRE (Mean Radial Error)**: Average pixel distance between predicted and ground truth keypoints (lower is better)
- **SDR (Successful Detection Rate)**: Percentage of keypoints detected within threshold:
  - SDR@2.0mm, SDR@2.5mm, SDR@3.0mm, SDR@4.0mm (higher is better)

### Target Performance Goals

| Model | MRE (pixels) | SDR@2mm (%) | SDR@3mm (%) | Status |
|-------|--------------|-------------|-------------|--------|
| U-Net | < 30 | > 70 | > 85 | 🔄 Training |
| ResNet-Keypoint | < 25 | > 75 | > 88 | 📋 Planned |
| Keypoint R-CNN | < 20 | > 80 | > 90 | 📋 Planned |
| MAHT-Net | < 15 | > 85 | > 92 | 📋 Planned |

*Note: These are target goals. Actual results will establish the first baselines on this dataset.*

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data understanding & exploration |
| Phase 2 | ⏳ Ready | Model implementation (U-Net, ResNet) |
| Phase 3 | ⏳ Pending | Advanced models (R-CNN, MAHT-Net) |
| Phase 4 | ⏳ Pending | Evaluation & comparison |
| Phase 5 | ⏳ Pending | Paper writing & submission |

**Current Focus**: Training baseline models to establish first performance benchmarks

---

## 🎯 Research Goals

1. **Establish Baselines**: Create the first comprehensive baseline results on this dataset
2. **Model Comparison**: Compare 4 different architectures for vertebral landmark detection
3. **Clinical Validation**: Achieve clinically-relevant accuracy (MRE < 3mm, Grade accuracy > 90%)
4. **Open Source**: Release code and pretrained models for research community

---

## 📈 Expected Results

Target performance metrics:

| Model | MRE (mm) | SDR@2mm (%) | Grade Accuracy (%) |
|-------|----------|-------------|-------------------|
| U-Net | ~4.0 | ~70 | ~85 |
| ResNet | ~3.5 | ~75 | ~87 |
| Keypoint R-CNN | ~3.0 | ~80 | ~90 |
| MAHT-Net | ~2.5 | ~85+ | ~92+ |

*Note: These are target goals. Actual results will establish the baseline.*

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

# Train full model in cloud
python train.py --model unet --epochs 50 --batch-size 16 --experiment-name production_v1

# Compare results
python scripts/compare_experiments.py --model unet
```