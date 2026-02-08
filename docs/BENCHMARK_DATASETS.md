# Benchmark Datasets for Spine Analysis Research

This document provides a comprehensive overview of publicly available benchmark datasets for spine analysis, vertebra segmentation, landmark detection, and spondylolisthesis research. These datasets enable comparison of our MAHT-Net model with state-of-the-art methods to facilitate publication in high-impact journals.

---

## ✅ Selected Benchmark: BUU-LSPINE

**Decision Date:** February 8, 2026

After careful evaluation of all available benchmark datasets, **BUU-LSPINE** has been selected as the primary benchmark for evaluating MAHT-Net.

### Why BUU-LSPINE Was Selected

| Criterion | BUU-LSPINE | Other Datasets | Verdict |
|-----------|------------|----------------|--------|
| **Annotation Style** | Vertebral corner points (4 corners/vertebra) | SpineWeb: corner points, VinDr: bounding boxes | ✅ **Perfect match** with our approach |
| **Clinical Task** | Spondylolisthesis detection + Meyerding grading | VinDr: spondylolisthesis (bbox only), SpineWeb: scoliosis | ✅ **Identical task** |
| **Modality** | X-ray (AP + Lateral views) | All X-ray datasets match | ✅ **Same modality** |
| **Dataset Size** | 7,200 images (3,600 patients) | VinDr: 10,466, SpineWeb: 609 | ✅ **Substantial size** |
| **Access** | Open access, no registration | VinDr: requires DUA, SpineWeb: research license | ✅ **Easy access** |
| **Published Baselines** | YOLOv5, ResNet152V2, DenseNet201, SVM | Limited or different tasks | ✅ **Direct comparison** |
| **Metrics** | Mean Error Distance (mm), Classification Accuracy | Different metrics | ✅ **Compatible metrics** |

### Key Advantages for MAHT-Net Evaluation

1. **Apples-to-Apples Comparison**: Same annotation format (corner points) enables direct keypoint detection comparison
2. **Clinical Relevance**: Dataset specifically designed for spondylolisthesis - our exact use case
3. **Reproducibility**: Open access dataset with published baselines and clear methodology
4. **Dual Views**: AP + Lateral views allow comprehensive evaluation
5. **Publication Ready**: Published in Applied Sciences (MDPI, 2023) with clear citation

### Comparison Targets

MAHT-Net will be benchmarked against these BUU-LSPINE baselines:

| Task | Baseline Model | Performance | Our Target |
|------|----------------|-------------|------------|
| Corner Point Detection | ResNet152V2 | 4.63mm error (AP) | **< 4.0mm** |
| Corner Point Detection | DenseNet201 | 4.91mm error (LA) | **< 4.5mm** |
| Spondylolisthesis Classification | SVM | 95.14% accuracy (AP) | **> 96%** |
| Vertebra Detection | YOLOv5 | 81.93% precision | **> 85%** |

---

## Table of Contents

1. [X-Ray Datasets](#x-ray-datasets)
   - [BUU-LSPINE (Thai Lumbar Spine Dataset)](#1-buu-lspine-thai-lumbar-spine-dataset) ✅ **SELECTED**
   - [SpineWeb Dataset 16 (AASCE Challenge)](#2-spineweb-dataset-16-aasce-challenge)
   - [VinDr-SpineXR](#3-vindr-spinexr)
2. [CT Datasets](#ct-datasets)
   - [VerSe Challenge Dataset](#4-verse-challenge-dataset)
   - [CTSpine1K](#5-ctspine1k)
   - [RSNA 2022 Cervical Spine Fracture Detection](#6-rsna-2022-cervical-spine-fracture-detection)
3. [MRI Datasets](#mri-datasets)
   - [Lumbar Spine MRI Dataset](#7-lumbar-spine-mri-dataset)
4. [Dataset Comparison Summary](#dataset-comparison-summary)
5. [Recommended Datasets for Our Research](#recommended-datasets-for-our-research)
6. [Citation Guidelines](#citation-guidelines)

---

## X-Ray Datasets

### 1. BUU-LSPINE (Thai Lumbar Spine Dataset) ✅ **SELECTED BENCHMARK**

**Overview:**
BUU-LSPINE is an **open Thai lumbar spine dataset specifically designed for spondylolisthesis detection**. Developed by Burapha University (Thailand) and the Korea Institute of Oriental Medicine, it provides **vertebral corner point annotations** and spondylolisthesis diagnosis - making it **extremely relevant** to our MAHT-Net research.

| Attribute | Details |
|-----------|---------||
| **Modality** | X-ray (Plain Film) |
| **Total Patients** | 3,600 patients |
| **Total Images** | 7,200 images (AP + LA views) |
| **Views** | Anteroposterior (AP) and Lateral (LA) |
| **Spondylolisthesis Cases** | 621 cases |
| **Annotations** | Vertebral corner points + Spondylolisthesis diagnosis |
| **Additional Data** | LSTV (Lumbosacral Transitional Vertebrae) labels |
| **Publication** | Applied Sciences (MDPI), 2023 |
| **License** | Open Access |

**🎯 Why This Dataset Was Selected:**
- **Same annotation style as our dataset:** Vertebral corner points (4 corners per vertebra)
- **Spondylolisthesis focus:** Includes diagnosis ground truth with Meyerding classification
- **Direct comparison possible:** Our MAHT-Net can be directly evaluated on this dataset
- **Large scale:** 3,600 patients is substantial for training/evaluation

**Annotation Details:**

*Vertebral Corner Points:*
- **AP View:** 10 vertebral edge lines (upper + lower edges for L1-L5)
- **LA View:** 11 vertebral edge lines (L1-L5 + S1a reference line)
- Each line contains left and right corner point coordinates
- Stored in CSV files per image

*Spondylolisthesis Classes:*
| Class | Type | AP View Count | LA View Count |
|-------|------|---------------|---------------|
| 0 | Normal | Majority | Majority |
| 1 | Left Laterolisthesis | 61 | - |
| 2 | Right Laterolisthesis | Present | - |
| 3 | Anterolisthesis | - | Present |
| 4 | Retrolisthesis | - | 194 |

*LSTV Classes:*
| Class | Description |
|-------|-------------|
| 0 | Normal (5 lumbar vertebrae) |
| 1 | Sacralization (4 lumbar - L5 fused with sacrum) |
| 2 | Lumbarization (6 lumbar - S1 acts as lumbar) |

**Data Structure:**
```
BUU-LSPINE/
├── [patient_id]_AP.jpg          # Anteroposterior view image
├── [patient_id]_AP.csv          # Corner points + diagnosis (AP)
├── [patient_id]_LA.jpg          # Lateral view image
└── [patient_id]_LA.csv          # Corner points + diagnosis + LSTV (LA)
```

**CSV Format (AP View):**
| Row | Line | Left X | Left Y | Right X | Right Y | Class |
|-----|------|--------|--------|---------|---------|-------|
| 1 | L1a (upper) | x1 | y1 | x2 | y2 | - |
| 2 | L1b (lower) | x1 | y1 | x2 | y2 | Spondy class |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | L5b (lower) | x1 | y1 | x2 | y2 | Spondy class |

**Baseline Results (from paper):**

*Lumbar Vertebrae Detection:*
| Model | AP Precision | LA Precision | AP Recall | LA Recall |
|-------|--------------|--------------|-----------|-----------||
| **YOLOv5** | **81.93%** | **83.45%** | **95.82%** | **95.72%** |
| MobileNetV1 | 78.21% | 79.34% | 93.45% | 92.18% |
| ResNet50V2 | 80.12% | 81.23% | 94.67% | 94.23% |

*Corner Point Extraction (Avg Error Distance):*
| Model | AP View (mm) | LA View (mm) |
|-------|--------------|---------------|
| **ResNet152V2** | **4.63** | 5.12 |
| **DenseNet201** | 4.89 | **4.91** |
| EfficientNetB0 | 5.23 | 5.45 |

*Spondylolisthesis Prediction Accuracy:*
| Classifier | AP View | LA View |
|------------|---------|--------|
| **SVM** | **95.14%** | **92.26%** |
| Random Forest | 93.45% | 90.12% |
| KNN | 91.23% | 88.56% |

**Access:**
| Resource | URL |
|----------|-----|
| Dataset Portal | https://services.informatics.buu.ac.th/spine/ |
| Paper (MDPI) | https://www.mdpi.com/2076-3417/13/15/8646 |
| DOI | https://doi.org/10.3390/app13158646 |

**Relevance to Our Work:** ⭐⭐⭐ **SELECTED AS PRIMARY BENCHMARK** ✅
- **Identical task:** Spondylolisthesis detection from X-ray with corner points
- **Direct comparison:** Can benchmark MAHT-Net against YOLOv5, ResNet, DenseNet
- **Same annotation approach:** Vertebral corner points
- **Published baselines:** Clear metrics to compare against
- **Status:** Selected for MAHT-Net evaluation (February 2026)

**Key Publications:**
```bibtex
@article{klinwichit2023buulspine,
  title={BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis Detection},
  author={Klinwichit, Podchara and Yookwan, Watcharaphong and Limchareon, Sornsupha 
          and Chinnasarn, Krisana and Jang, Jun-Su and Onuean, Athita},
  journal={Applied Sciences},
  volume={13},
  number={15},
  pages={8646},
  year={2023},
  publisher={MDPI},
  doi={10.3390/app13158646}
}
```

---

### 2. SpineWeb Dataset 16 (AASCE Challenge)

**Overview:**
The SpineWeb Dataset 16 is the primary dataset used in the **MICCAI 2019 AASCE (Accurate Automated Spinal Curvature Estimation) Challenge**. It focuses on scoliosis assessment through Cobb angle estimation from anterior-posterior (AP) X-ray images.

| Attribute | Details |
|-----------|---------|
| **Modality** | X-ray (AP view) |
| **Focus** | Adolescent Idiopathic Scoliosis (AIS) |
| **Total Images** | 609 images (training + test) |
| **Annotations** | 4 corner points per vertebra (68 landmarks for 17 vertebrae) |
| **Task** | Vertebra landmark detection, Cobb angle estimation |
| **Challenge** | MICCAI 2019 AASCE Challenge |

**Annotation Format:**
- Each vertebra annotated with 4 corner points
- Point ordering: top-left, top-right, bottom-right, bottom-left
- Format: `[x1, y1, x2, y2, x3, y3, x4, y4]`
- Stored in XML files (one per image)

**Curated Annotations Available:**
The original annotations contain label noise (shifts, flips, swaps). Curated/corrected annotations are available addressing:
- Upward/Downward shift of vertebral body annotations
- Flipping of corner points and top/end plates
- Swap of vertebral annotations
- Overlapping vertebrae
- Corner point duplicates
- Anatomically implausible positions

**Access:**
| Resource | URL |
|----------|-----|
| SpineWeb Portal | http://spineweb.digitalimaginggroup.ca/ |
| Curated Annotations | https://zenodo.org/records/4413665 |
| DOI (Curated) | 10.5281/zenodo.4413665 |

**Challenge Winners (2019):**
1. **1st Place:** Seg4Reg Networks (Tencent YouTu Lab)
2. **2nd Place:** Accurate Automated Keypoint Detections (iFLYTEK Research)
3. **3rd Place:** Spine Centerline Extraction with Cascaded Neural Networks (Erasmus MC)

**Relevance to Our Work:** ⭐⭐⭐ **Highly Relevant**
- Contains vertebra corner point annotations similar to our spondylolisthesis dataset
- Direct keypoint detection comparison possible
- Established benchmark with published baselines

**Key Publications:**
```bibtex
@inproceedings{wu2017boostnet,
  title={Automatic Landmark Estimation for Adolescent Idiopathic Scoliosis 
         Assessment Using BoostNet},
  author={Wu, Hongbo and Bailey, Chris and Rasoulinejad, Parham and Li, Shuo},
  booktitle={MICCAI},
  pages={127--135},
  year={2017}
}
```

---

### 3. VinDr-SpineXR

**Overview:**
VinDr-SpineXR is the **largest publicly available spine X-ray dataset** with bounding box annotations for multiple spinal lesions, **including spondylolisthesis**. This makes it particularly valuable for our research.

| Attribute | Details |
|-----------|---------|
| **Modality** | X-ray (cervical, thoracic, lumbar) |
| **Total Images** | 10,466 images from 5,000 studies |
| **Training Set** | 8,389 images (4,000 studies) |
| **Test Set** | 2,077 images (1,000 studies) |
| **Annotations** | Bounding boxes with 13 lesion types |
| **Annotators** | 3 experienced radiologists |
| **Publication** | MICCAI 2021 |

**13 Lesion Types Annotated:**
| # | Lesion Type | # | Lesion Type |
|---|-------------|---|-------------|
| 1 | Ankylosis | 8 | **Spondylolisthesis** ⭐ |
| 2 | Disc space narrowing | 9 | Subchondral sclerosis |
| 3 | Enthesophytes | 10 | Surgical implant |
| 4 | Foraminal stenosis | 11 | Vertebral collapse |
| 5 | Fracture | 12 | Foreign body |
| 6 | Osteophytes | 13 | Other lesions |
| 7 | Sclerotic lesion | - | No finding |

**Data Organization:**
```
├── annotations/
│   ├── train.csv
│   └── test.csv
├── train_images/
│   └── *.dicom
└── test_images/
    └── *.dicom
```

**Annotation CSV Format:**
| Column | Description |
|--------|-------------|
| `image_id` | Unique image identifier |
| `series_id` | Series identifier |
| `study_id` | Study identifier |
| `rad_id` | Radiologist ID (rad1, rad2, rad3) |
| `lesion_type` | One of 13 lesions or "No finding" |
| `xmin`, `ymin`, `xmax`, `ymax` | Bounding box coordinates |

**Access:**
| Resource | URL |
|----------|-----|
| PhysioNet | https://physionet.org/content/vindr-spinexr/1.0.0/ |
| Project Website | https://vindr.ai/datasets/spinexr |
| DOI | https://doi.org/10.13026/q45h-5h59 |
| License | PhysioNet Restricted Health Data License 1.5.0 |

**⚠️ Access Requirements:**
- Must register on PhysioNet
- Must sign Data Use Agreement (DUA)

**Relevance to Our Work:** ⭐⭐⭐ **Extremely Relevant**
- **Contains spondylolisthesis annotations** - direct comparison possible
- Same modality (X-ray) as our dataset
- Published in MICCAI 2021

**Key Publications:**
```bibtex
@inproceedings{nguyen2021vindr,
  title={VinDr-SpineXR: A deep learning framework for spinal lesions 
         detection and classification from radiographs},
  author={Nguyen, Hieu T and Pham, Hieu H and Nguyen, Nghia T and 
          Nguyen, Ha Q and Huynh, Thang Q and Dao, Minh and Vu, Van},
  booktitle={MICCAI},
  year={2021}
}
```

---

## CT Datasets

### 4. VerSe Challenge Dataset

**Overview:**
VerSe (Large Scale Vertebrae Segmentation Challenge) is the **largest public CT spine dataset**, used in MICCAI 2019 and 2020 challenges. It provides both segmentation masks and centroid annotations.

| Attribute | Details |
|-----------|---------|
| **Modality** | CT |
| **Total Scans** | 374 scans from 355 patients |
| **Coverage** | Multi-detector, multi-site |
| **Annotations** | Segmentation masks + centroid coordinates |
| **Vertebrae Labeled** | C1-C7, T1-T13, L1-L6 |
| **License** | CC BY-SA 4.0 (fully open-sourced) |
| **Challenges** | MICCAI 2019 & 2020 |

**Data Structure (BIDS format):**
```
training/rawdata/sub-verse000/
    sub-verse000_dir-orient_ct.nii.gz           # CT image

training/derivatives/sub-verse000/
    sub-verse000_dir-orient_seg-vert_msk.nii.gz  # Segmentation mask
    sub-verse000_dir-orient_seg-subreg_ctd.json  # Centroid coordinates
    sub-verse000_dir-orient_seg-vert_snp.png     # Preview
```

**Vertebral Label Mapping:**
| Label Range | Vertebrae |
|-------------|-----------|
| 1-7 | Cervical (C1-C7) |
| 8-19 | Thoracic (T1-T12) |
| 20-25 | Lumbar (L1-L6) |
| 28 | Additional T13 |

**Access:**
| Resource | URL |
|----------|-----|
| GitHub | https://github.com/anjany/verse |
| VerSe'19 (OSF) | https://osf.io/nqjyw/ |
| VerSe'20 (OSF) | https://osf.io/t98fz/ |

**Direct Downloads (WGET):**
```bash
# VerSe'19
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip

# VerSe'20
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip
wget https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip
```

**Relevance to Our Work:** ⭐⭐ **Relevant**
- Can adapt centroid detection methods to CT modality
- Well-established benchmark with published baselines
- Cross-modality comparison possible

**Key Publications:**
```bibtex
@article{sekuboyina2021verse,
  title={VerSe: A Vertebrae Labelling and Segmentation Benchmark for 
         Multi-detector CT Images},
  author={Sekuboyina, Anjany and others},
  journal={Medical Image Analysis},
  year={2021},
  doi={10.1016/j.media.2021.102166}
}

@article{loffler2020vertebral,
  title={A Vertebral Segmentation Dataset with Fracture Grading},
  author={L{\"o}ffler, Maximilian and others},
  journal={Radiology: Artificial Intelligence},
  year={2020},
  doi={10.1148/ryai.2020190138}
}
```

---

### 5. CTSpine1K

**Overview:**
CTSpine1K is a large-scale CT spine dataset combining data from multiple public sources, designed for vertebrae segmentation. **Accepted at MICCAI 2024 Open Data** for oral presentation.

| Attribute | Details |
|-----------|---------|
| **Modality** | CT |
| **Total Volumes** | 1,005 CT volumes |
| **Total Slices** | >500,000 labeled slices |
| **Total Vertebrae** | >11,000 vertebrae |
| **License** | CC-BY-NC-SA |
| **Format** | NIfTI |

**Data Sources:**
| Source | # Scans | Description |
|--------|---------|-------------|
| COLONOG | 825 | CT colonography trial |
| HNSCC-3DCT-RT | - | Head-and-neck cancer CTs |
| MSD T10 | 201 | Medical Segmentation Decathlon (liver) |
| COVID-19 | 40 | Chest CTs from COVID patients |

**Access:**
| Resource | URL |
|----------|-----|
| GitHub | https://github.com/MIRACLE-Center/CTSpine1K |
| Hugging Face | https://huggingface.co/datasets/alexanderdann/CTSpine1K |
| Google Drive | https://drive.google.com/drive/folders/1Acyuu7ZmbjnS4mkJRdiUfkXx5SBta4EM |
| XNAT (AFRICAI) | https://xnat.bmia.nl/data/archive/projects/africai_miccai2024_ctspine1k |

**Relevance to Our Work:** ⭐ **Moderately Relevant**
- CT modality differs from X-ray
- Useful for demonstrating method generalization
- Large-scale dataset for benchmarking

**Key Publications:**
```bibtex
@article{deng2021ctspine1k,
  title={CTSpine1K: A large-scale dataset for spinal vertebrae 
         segmentation in computed tomography},
  author={Deng, Yang and Wang, Ce and Hui, Yuan and others},
  journal={arXiv preprint arXiv:2105.14711},
  year={2021}
}
```

---

### 6. RSNA 2022 Cervical Spine Fracture Detection

**Overview:**
A large-scale Kaggle competition dataset for cervical spine fracture detection from CT scans.

| Attribute | Details |
|-----------|---------|
| **Modality** | CT (DICOM format) |
| **Total Scans** | ~2,000 patient scans |
| **Focus** | Cervical spine (C1-C7) fractures |
| **Annotations** | Patient-level + vertebra-level fracture labels |
| **Additional** | Pixel-level segmentations (subset) |
| **Size** | 343.51 GB |
| **Files** | 713,010 files |

**Data Files:**
| File | Description |
|------|-------------|
| `train.csv` | Metadata with fracture labels |
| `train_bounding_boxes.csv` | Bounding boxes (subset) |
| `segmentations/` | NIfTI segmentation masks |
| `train_images/` | DICOM image folders |

**Segmentation Labels:**
- 1-7: C1-C7 (cervical vertebrae)
- 8-19: T1-T12 (thoracic vertebrae)

**Access:**
| Resource | URL |
|----------|-----|
| Kaggle | https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/data |
| Download Command | `kaggle competitions download -c rsna-2022-cervical-spine-fracture-detection` |

**Relevance to Our Work:** ⭐ **Moderately Relevant**
- Focuses on fractures, not spondylolisthesis
- CT modality different from X-ray
- Large dataset for vertebra localization

---

## MRI Datasets

### 7. Lumbar Spine MRI Dataset

**Overview:**
Clinical MRI dataset of patients with symptomatic back pain, focusing on lumbar spine and intervertebral discs.

| Attribute | Details |
|-----------|---------|
| **Modality** | MRI |
| **Patients** | 515 patients |
| **Total Slices** | 48,345 MRI slices |
| **Views** | Sagittal and axial |
| **Resolution** | 320×320 pixels (mostly) |
| **Pixel Precision** | 12-bit |
| **Focus** | L3-L5 vertebrae, intervertebral discs |
| **License** | CC BY 4.0 |
| **Size** | 5.83 GB |

**Technical Specifications:**
| Parameter | Value |
|-----------|-------|
| Slice thickness | 4 mm (axial) |
| Slice spacing | 4.4 mm |
| Pixel spacing | 0.6875 mm |
| Patient position | Head-First-Supine / Feet-First-Supine |

**Access:**
| Resource | URL |
|----------|-----|
| Mendeley Data | https://data.mendeley.com/datasets/k57fr854j2/2 |
| DOI | 10.17632/k57fr854j2.2 |

**Citation:**
```bibtex
@dataset{sudirman2019lumbar,
  author = {Sudirman, Sud and Al Kafri, Ala and Natalia, Friska and 
            Meidia, Hira and Afriliana, Nunik and Al-Rashdan, Wasfi and 
            Bashtawi, Mohammad and Al-Jumaily, Mohammed},
  title = {Lumbar Spine MRI Dataset},
  year = {2019},
  publisher = {Mendeley Data},
  doi = {10.17632/k57fr854j2.2}
}
```

**Relevance to Our Work:** ⭐ **Low-Moderate Relevance**
- Different modality (MRI vs X-ray)
- Useful for multi-modality studies

---

## Dataset Comparison Summary

| Dataset | Modality | Size | Keypoints | Spondylolisthesis | License | Relevance |
|---------|----------|------|-----------|-------------------|---------|-----------|
| **BUU-LSPINE** ✅ | X-ray | 7,200 | ✅ Corner pts | ✅ | Open Access | ⭐⭐⭐ **SELECTED** |
| **SpineWeb 16** | X-ray | 609 | ✅ 68 pts | ❌ | Research | ⭐⭐⭐ |
| **VinDr-SpineXR** | X-ray | 10,466 | ❌ (bbox) | ✅ | Restricted | ⭐⭐⭐ |
| **VerSe** | CT | 374 | ✅ Centroids | ❌ | CC BY-SA 4.0 | ⭐⭐ |
| **CTSpine1K** | CT | 1,005 | ✅ Masks | ❌ | CC-BY-NC-SA | ⭐ |
| **RSNA Cervical** | CT | ~2,000 | ✅ Seg | ❌ | Competition | ⭐ |
| **Lumbar MRI** | MRI | 48,345 | ❌ | ❌ | CC BY 4.0 | ⭐ |

---

## Recommended Datasets for Our Research

### Primary Benchmarks (Highly Recommended)

#### 1. BUU-LSPINE ⭐⭐⭐ ✅ **[SELECTED]**
**Why:** Identical task - spondylolisthesis detection with vertebral corner points

**Advantages:**
- **Same annotation approach** as our dataset (corner points)
- **Same clinical task** (spondylolisthesis detection)
- **Same modality** (X-ray lumbar spine)
- **Published baselines** in Applied Sciences 2023
- **Open access** - no registration required
- **Large scale** (3,600 patients, 7,200 images)

**Evaluation Metrics:**
- Mean Error Distance (mm) for corner point detection
- Spondylolisthesis classification accuracy
- Precision/Recall for vertebra detection

#### 2. VinDr-SpineXR ⭐⭐⭐
**Why:** Contains actual spondylolisthesis annotations in X-ray format

**Advantages:**
- Direct comparison for spondylolisthesis detection
- Same modality (X-ray) as our dataset
- Published in top venue (MICCAI 2021)
- Large-scale (10,466 images)

**Evaluation Metrics:**
- mAP (mean Average Precision) for lesion detection
- Sensitivity/Specificity for spondylolisthesis
- AUC-ROC curves

#### 3. SpineWeb Dataset 16 (AASCE) ⭐⭐⭐
**Why:** Contains corner point annotations similar to our approach

**Advantages:**
- Direct keypoint detection comparison
- Established benchmark with challenge baselines
- MICCAI challenge dataset
- Curated annotations available

**Evaluation Metrics:**
- Mean Radial Error (MRE) for landmarks
- Successful Detection Rate (SDR) at different thresholds (2mm, 4mm, 10mm)
- Cobb angle estimation error (MAE, SMAPE)

### Secondary Benchmarks (Recommended)

#### 4. VerSe ⭐⭐
**Why:** Well-established CT benchmark for vertebra detection

**Advantages:**
- Demonstrates cross-modality generalization
- Extensive published baselines
- Open license (CC BY-SA 4.0)
- Large research community

**Evaluation Metrics:**
- Dice Score for segmentation
- Identification Rate (IR)
- Localization Distance Error

---

## Citation Guidelines

When using these datasets, ensure proper citation:

### BUU-LSPINE
```bibtex
@article{klinwichit2023buulspine,
  title={BUU-LSPINE: A Thai Open Lumbar Spine Dataset for Spondylolisthesis Detection},
  author={Klinwichit, Podchara and Yookwan, Watcharaphong and Limchareon, Sornsupha 
          and Chinnasarn, Krisana and Jang, Jun-Su and Onuean, Athita},
  journal={Applied Sciences},
  volume={13},
  number={15},
  pages={8646},
  year={2023},
  publisher={MDPI},
  doi={10.3390/app13158646}
}
```

### VinDr-SpineXR
```bibtex
@inproceedings{nguyen2021vindr,
  title={VinDr-SpineXR: A deep learning framework for spinal lesions 
         detection and classification from radiographs},
  author={Nguyen, Hieu T and Pham, Hieu H and Nguyen, Nghia T and 
          Nguyen, Ha Q and Huynh, Thang Q and Dao, Minh and Vu, Van},
  booktitle={International Conference on Medical Image Computing and 
             Computer-Assisted Intervention (MICCAI)},
  year={2021}
}

@misc{pham2021vindr_dataset,
  author = {Pham, Hieu Huy and Nguyen Trung, Hieu and Nguyen, Ha Quy},
  title = {VinDr-SpineXR: A large annotated medical image dataset for 
           spinal lesions detection and classification from radiographs},
  year = {2021},
  publisher = {PhysioNet},
  doi = {10.13026/q45h-5h59}
}
```

### SpineWeb / AASCE
```bibtex
@inproceedings{wu2017boostnet,
  title={Automatic Landmark Estimation for Adolescent Idiopathic Scoliosis 
         Assessment Using BoostNet},
  author={Wu, Hongbo and Bailey, Chris and Rasoulinejad, Parham and Li, Shuo},
  booktitle={International Conference on Medical Image Computing and 
             Computer-Assisted Intervention (MICCAI)},
  pages={127--135},
  year={2017}
}

@misc{kordon2021curated,
  author = {Kordon, Florian},
  title = {Curated annotations for SpineWeb Dataset 16},
  year = {2021},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.4413665}
}
```

### VerSe
```bibtex
@article{sekuboyina2021verse,
  title={VerSe: A Vertebrae Labelling and Segmentation Benchmark for 
         Multi-detector CT Images},
  author={Sekuboyina, Anjany and others},
  journal={Medical Image Analysis},
  volume={73},
  pages={102166},
  year={2021},
  doi={10.1016/j.media.2021.102166}
}
```

### CTSpine1K
```bibtex
@article{deng2021ctspine1k,
  title={CTSpine1K: A large-scale dataset for spinal vertebrae 
         segmentation in computed tomography},
  author={Deng, Yang and Wang, Ce and Hui, Yuan and others},
  journal={arXiv preprint arXiv:2105.14711},
  year={2021}
}
```

---

## Experimental Design for Publication

### Recommended Evaluation Protocol

| Metric | BUU-LSPINE | SpineWeb | VinDr-SpineXR | VerSe |
|--------|------------|----------|---------------|-------|
| MRE (mm) | ✅ | ✅ | - | - |
| SDR (%) | ✅ | ✅ | - | - |
| mAP | - | - | ✅ | - |
| Dice | - | - | - | ✅ |
| ID Rate | - | - | - | ✅ |
| AUC-ROC | - | - | ✅ | - |
| Classification Acc | ✅ | - | - | - |

### Publication Strategy

1. **Primary Comparison:** BUU-LSPINE (most similar task) + SpineWeb (established benchmark)
2. **Secondary Comparison:** VinDr-SpineXR for spondylolisthesis detection
3. **Extended Comparison:** Include VerSe for CT generalization
4. **Ablation Studies:** On our own spondylolisthesis dataset

### Comparison Methods to Include

| Method | Year | Venue | Dataset |
|--------|------|-------|---------|
| YOLOv5 | 2023 | Applied Sciences | BUU-LSPINE |
| ResNet152V2 | 2023 | Applied Sciences | BUU-LSPINE |
| DenseNet201 | 2023 | Applied Sciences | BUU-LSPINE |
| BoostNet | 2017 | MICCAI | SpineWeb |
| Seg4Reg | 2019 | AASCE | SpineWeb |
| MVC-Net | 2018 | MedIA | SpineWeb |
| VinDr Framework | 2021 | MICCAI | VinDr-SpineXR |

---

## Access Checklist

- [ ] **BUU-LSPINE:** Download from https://services.informatics.buu.ac.th/spine/ (open access)
- [ ] **SpineWeb:** Request access at http://spineweb.digitalimaginggroup.ca/
- [ ] **SpineWeb Curated:** Download from https://zenodo.org/records/4413665
- [ ] **VinDr-SpineXR:** Sign DUA at PhysioNet
- [ ] **VerSe:** Direct download (open access)
- [ ] **CTSpine1K:** Download from Hugging Face or Google Drive
- [ ] **RSNA Cervical:** Kaggle account required
- [ ] **Lumbar MRI:** Direct download from Mendeley

---

*Document created: February 7, 2026*  
*Last updated: February 7, 2026*  
*Author: PhD Research Team*
