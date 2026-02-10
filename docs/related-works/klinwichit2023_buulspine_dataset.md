# BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis Detection

## Paper Information

| Field | Details |
|-------|---------|
| **Title** | BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis Detection |
| **Authors** | Podchara Klinwichit, Watcharaphong Yookwan, Sornsupha Limchareon, Krisana Chinnasarn, Jun-Su Jang, Athita Onuean |
| **Journal** | Applied Sciences (MDPI) |
| **Year** | 2023 |
| **Volume/Issue** | 13(15), Article 8646 |
| **DOI** | [10.3390/app13158646](https://doi.org/10.3390/app13158646) |
| **Publisher** | MDPI |
| **License** | Open Access (CC BY 4.0) |

---

## Summary

### Problem Statement

Spondylolisthesis diagnosis requires accurate identification of vertebral positions and measurement of vertebral slip. Existing public datasets for spine analysis either:
- Focus on scoliosis rather than spondylolisthesis
- Use CT/MRI modalities rather than X-ray
- Lack vertebral corner point annotations needed for slip measurement
- Are not openly accessible to researchers

### Main Contribution

The authors introduce **BUU-LSPINE**, the first large-scale **open Thai lumbar spine X-ray dataset** specifically designed for spondylolisthesis detection, featuring:

1. **7,200 X-ray images** from 3,600 Thai patients
2. **Dual view coverage**: Anteroposterior (AP) and Lateral (LA) views
3. **Vertebral corner point annotations** for L1-L5 (+ S1 reference in LA)
4. **Spondylolisthesis diagnosis labels** with type classification
5. **LSTV (Lumbosacral Transitional Vertebrae)** annotations

### Dataset Specifications

| Attribute | Value |
|-----------|-------|
| **Total Patients** | 3,600 |
| **Total Images** | 7,200 (AP + LA pairs) |
| **Spondylolisthesis Cases** | 621 (17.25%) |
| **Normal Cases** | 2,979 (82.75%) |
| **Image Resolution** | ~2536×3006 pixels |
| **Annotation Format** | CSV (corner points + labels) |

### Annotation Schema

**AP View (10 rows per image):**
- 5 vertebrae × 2 endplates (superior + inferior)
- Format: `x_left, y_left, x_right, y_right, spondylolisthesis_class`

**LA View (11 rows per image):**
- Same as AP + S1 superior endplate reference
- Additional LSTV classification

**Spondylolisthesis Classes:**
| Class | Type | Description |
|-------|------|-------------|
| 0 | Normal | No slip |
| 1 | Left Laterolisthesis | Lateral slip to the left (AP only) |
| 2 | Right Laterolisthesis | Lateral slip to the right (AP only) |
| 3 | Anterolisthesis | Forward slip (LA only) |
| 4 | Retrolisthesis | Backward slip (LA only) |

### Baseline Methods & Results

The paper establishes baselines for three tasks:

#### Task 1: Vertebrae Detection

| Model | AP Precision | LA Precision | AP Recall | LA Recall |
|-------|--------------|--------------|-----------|-----------|
| **YOLOv5** | **81.93%** | **83.45%** | **95.82%** | **95.72%** |
| MobileNetV1 | 78.21% | 79.34% | 93.45% | 92.18% |
| ResNet50V2 | 80.12% | 81.23% | 94.67% | 94.23% |

#### Task 2: Corner Point Extraction (Mean Error Distance)

| Model | AP View (mm) | LA View (mm) |
|-------|--------------|--------------|
| **ResNet152V2** | **4.63** | 5.12 |
| **DenseNet201** | 4.89 | **4.91** |
| EfficientNetB0 | 5.23 | 5.45 |
| InceptionV3 | 5.01 | 5.34 |

#### Task 3: Spondylolisthesis Classification

| Classifier | AP Accuracy | LA Accuracy |
|------------|-------------|-------------|
| **SVM** | **95.14%** | **92.26%** |
| Random Forest | 93.45% | 90.12% |
| KNN | 91.23% | 88.56% |
| Decision Tree | 89.78% | 86.34% |

### Key Findings

1. **YOLOv5** achieved best vertebrae detection with >95% recall
2. **ResNet152V2** achieved lowest corner point error (4.63mm for AP)
3. **SVM classifier** achieved highest spondylolisthesis classification accuracy
4. **LA view** slightly more challenging than AP for most tasks
5. **Two-stage pipeline** (detection → corner extraction → classification) achieves >95% accuracy

---

## Article Input (For Your Paper)

### How to Cite This Work

This is the **primary dataset paper** - you should cite it as the benchmark dataset:

```
We evaluate our proposed method on the BUU-LSPINE dataset [X], a large-scale 
open Thai lumbar spine X-ray dataset containing 7,200 images from 3,600 patients 
with vertebral corner point annotations and spondylolisthesis diagnosis labels. 
This dataset provides established baselines using YOLOv5, ResNet152V2, and SVM, 
enabling direct comparison with our MAHT-Net approach.
```

### Positioning Against Your MAHT-Net Work

| Aspect | BUU-LSPINE Baselines | Your MAHT-Net |
|--------|----------------------|---------------|
| **Pipeline** | Multi-stage (detect → extract → classify) | End-to-end keypoint prediction |
| **Corner Point Method** | ResNet152V2 regression | Multi-scale attention heatmap |
| **Best Error** | 4.63mm (AP), 4.91mm (LA) | Target: <4.0mm |
| **Classification** | Separate SVM classifier | Integrated with detection |
| **Architecture** | Transfer learning (ImageNet) | Task-specific design |

### Key Baselines to Beat

| Task | Baseline | Performance | Your Target |
|------|----------|-------------|-------------|
| Corner Point Detection | ResNet152V2 | 4.63mm error (AP) | **< 4.0mm** |
| Corner Point Detection | DenseNet201 | 4.91mm error (LA) | **< 4.5mm** |
| Spondylolisthesis Classification | SVM | 95.14% accuracy | **> 96%** |
| Vertebra Detection | YOLOv5 | 81.93% precision | **> 85%** |

### Limitations to Address

The BUU-LSPINE baseline approach has several limitations your MAHT-Net can address:

1. **Multi-stage Pipeline**: Detection → Corner extraction → Classification introduces error accumulation
2. **Generic Architectures**: ResNet/DenseNet not specifically designed for keypoint detection
3. **Separate Classification**: SVM classifier trained separately, not end-to-end
4. **No Attention Mechanism**: Lacks focus on clinically relevant regions

### Suggested Related Works / Dataset Section

> **Dataset**: We evaluate our method on the BUU-LSPINE dataset [X], a publicly available Thai lumbar spine X-ray dataset comprising 7,200 images from 3,600 patients. Each image is annotated with vertebral corner point coordinates for L1-L5 (10 points for AP view, 11 for LA view including S1 reference) and spondylolisthesis diagnosis labels. The dataset includes 621 spondylolisthesis cases (17.25%) across multiple types: laterolisthesis (AP view), anterolisthesis and retrolisthesis (LA view). We follow the standard train/validation/test split and compare against published baselines including ResNet152V2 (4.63mm corner point error) and SVM classification (95.14% accuracy). Our proposed MAHT-Net achieves [YOUR RESULTS], demonstrating [IMPROVEMENT] over the multi-stage baseline approach.

### Suggested Methods Comparison Paragraph

> **Comparison with BUU-LSPINE Baselines**: The original BUU-LSPINE paper [X] established baselines using a three-stage pipeline: (1) YOLOv5 for vertebrae detection (81.93% precision), (2) ResNet152V2 for corner point extraction (4.63mm error), and (3) SVM for spondylolisthesis classification (95.14% accuracy). This multi-stage approach, while effective, suffers from error propagation between stages and requires separate training for each component. In contrast, our MAHT-Net employs an end-to-end architecture with multi-scale attention that directly predicts corner point coordinates, eliminating intermediate detection steps and enabling joint optimization of localization and classification objectives.

### BibTeX Citation

```bibtex
@article{klinwichit2023buulspine,
  author    = {Klinwichit, Podchara and Yookwan, Watcharaphong and 
               Limchareon, Sornsupha and Chinnasarn, Krisana and 
               Jang, Jun-Su and Onuean, Athita},
  title     = {BUU-LSPINE: A Thai Open Lumbar Spine Dataset for 
               Spondylolisthesis Detection},
  journal   = {Applied Sciences},
  volume    = {13},
  number    = {15},
  pages     = {8646},
  year      = {2023},
  publisher = {MDPI},
  doi       = {10.3390/app13158646}
}
```

---

## Relevance to Your Research

| Relevance Factor | Rating | Notes |
|------------------|--------|-------|
| **Primary Dataset** | ⭐⭐⭐⭐⭐ | **SELECTED** as your benchmark dataset |
| **Annotation Match** | ⭐⭐⭐⭐⭐ | Corner points - exact format needed |
| **Baseline Comparison** | ⭐⭐⭐⭐⭐ | Clear metrics to beat |
| **Open Access** | ⭐⭐⭐⭐⭐ | Freely available for research |
| **Clinical Task Match** | ⭐⭐⭐⭐⭐ | Spondylolisthesis - your exact task |

**Bottom Line**: This is your **primary benchmark paper**. You must cite it when describing the dataset and should compare your MAHT-Net results against **all three baseline tasks** (detection, corner extraction, classification) to demonstrate comprehensive improvement.

---

## Data Access

| Resource | URL |
|----------|-----|
| **Dataset Portal** | https://services.informatics.buu.ac.th/spine/ |
| **Paper (MDPI)** | https://www.mdpi.com/2076-3417/13/15/8646 |
| **DOI** | https://doi.org/10.3390/app13158646 |

---

## Technical Notes for Implementation

### Converting Annotations to MAHT-Net Format

BUU-LSPINE provides **2 points per endplate** (left and right corners). To get **4 corners per vertebra**:

```python
# For vertebra Li (e.g., L1):
# Row 2i: Superior endplate (Li_a)
# Row 2i+1: Inferior endplate (Li_b)

vertebra_corners = {
    'top_left': (row[2i].x_left, row[2i].y_left),
    'top_right': (row[2i].x_right, row[2i].y_right),
    'bottom_left': (row[2i+1].x_left, row[2i+1].y_left),
    'bottom_right': (row[2i+1].x_right, row[2i+1].y_right)
}
```

### Evaluation Metrics

Match the paper's evaluation metrics for fair comparison:
- **Mean Error Distance (mm)**: Euclidean distance between predicted and ground truth corner points
- **Precision/Recall**: For vertebra detection
- **Accuracy**: For spondylolisthesis classification

### Pixel-to-mm Conversion

The paper uses physical distance (mm) for corner point error. You'll need pixel spacing calibration - check if the dataset provides this or estimate from typical X-ray parameters.
