# Phase 1 Detailed Guide: Data Understanding & Exploration

**Project**: Spondylolisthesis MAHT-Net Research
**Duration**: Week 1 (7 days)
**Objective**: Thoroughly understand your dataset before building any models

---

## Table of Contents
1. [Overview & Purpose](#overview--purpose)
2. [Key Terminologies Explained](#key-terminologies-explained)
3. [Phase 1.1: Initial Data Exploration](#phase-11-initial-data-exploration)
4. [Phase 1.2: Visual Data Exploration](#phase-12-visual-data-exploration)
5. [Phase 1.3: Statistical Analysis](#phase-13-statistical-analysis)
6. [Phase 1.4: Data Quality & Validation](#phase-14-data-quality--validation)
5. [Expected Outputs](#expected-outputs)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview & Purpose

### Why Phase 1 is Critical

Before training any machine learning model, you **must** understand your data deeply. This phase answers:
- What exactly do my annotations represent?
- Are the images consistent in quality?
- How are keypoints distributed?
- Are there any data quality issues?
- What clinical patterns exist?

**Analogy**: This is like a chef inspecting all ingredients before cooking. You wouldn't cook without knowing what you have!

---

## Key Terminologies Explained

### 1. **Spondylolisthesis**
**Medical Definition**: A spinal condition where one vertebra slips forward over the vertebra below it.

**Clinical Importance**: 
- Affects 5-20% of adults
- Causes lower back pain and nerve compression
- Severity is graded using the Meyerding system

**In Your Project**: You're automating the detection and grading of this condition using AI.

---

### 2. **Vertebrae (L3, L4, L5, S1)**

**What are they?**
- **L3, L4, L5**: Lumbar vertebrae (lower back bones)
  - L = Lumbar
  - Numbers 3, 4, 5 = position from top to bottom
- **S1**: First sacral vertebra (base of spine)

**Visualization**:
```
    [L3] ← Vertebra 3
    [L4] ← Vertebra 4
    [L5] ← Vertebra 5
    [S1] ← Sacrum (first bone)
```

**Why These 4?**: Spondylolisthesis most commonly occurs between L4-L5 or L5-S1, so these are the critical vertebrae to analyze.

---

### 3. **Keypoints / Landmarks**

**Definition**: Specific anatomical points marked on each vertebra for precise measurements.

**Your Dataset Structure**:
- **4 vertebrae** (L3, L4, L5, S1)
- **4 corners per vertebra** (top-left, top-right, bottom-left, bottom-right)
- **Total: 16 keypoints per X-ray image**

**Visual Representation**:
```
Each vertebra looks like a rectangle with 4 corners:

    TL -------- TR     TL = Top-Left corner
    |          |       TR = Top-Right corner
    |  [L4]    |       BL = Bottom-Left corner
    |          |       BR = Bottom-Right corner
    BL -------- BR
```

**Keypoint Format**: `[x, y, visibility]`
- `x`: Horizontal position (0 = left edge)
- `y`: Vertical position (0 = top edge)
- `visibility`: 1 = visible, 0 = not visible/occluded

**Example from your data**:
```json
[[331, 28, 1], [396, 37, 1], [357, 1, 1], [405, 9, 1]]
     ↑    ↑   ↑
     x    y   visible
```

---

### 4. **Bounding Box**

**Definition**: A rectangular box that encloses a vertebra.

**Format**: `[x_min, y_min, x_max, y_max]`
- `x_min, y_min`: Top-left corner of the box
- `x_max, y_max`: Bottom-right corner of the box

**Example**:
```json
[331, 1, 405, 37]
  ↑   ↑   ↑   ↑
 left top right bottom
```

**Purpose**: 
- Quick object localization
- Computed from the 4 keypoints (smallest rectangle containing all points)
- Used in object detection models

---

### 5. **Annotation Format (JSON)**

**Your Data Structure**:
```json
{
  "boxes": [
    [331, 1, 405, 37],      // Bounding box for L3
    [309, 39, 392, 87],     // Bounding box for L4
    // ... more boxes
  ],
  "keypoints": [
    [[331, 28, 1], [396, 37, 1], [357, 1, 1], [405, 9, 1]],  // 4 corners of L3
    [[309, 73, 1], [377, 87, 1], [324, 39, 1], [392, 49, 1]], // 4 corners of L4
    // ... more keypoints
  ],
  "labels": [0, 0, 0, 0]   // Class labels (all vertebrae, so all 0)
}
```

**Understanding the Structure**:
- **boxes**: List of 4 bounding boxes (one per vertebra)
- **keypoints**: List of 4 sets of 4 corners (nested lists)
- **labels**: Class identifier (0 = vertebra class)

---

### 6. **Train/Val/Test Split**

**Purpose**: Divide data into separate sets for different purposes.

**Your Dataset Split**:
```
Total: 716 images
├── Train: 494 images (69%)  ← Learn patterns from these
├── Val:   206 images (29%)  ← Check performance during training
└── Test:   16 images (2%)   ← Final clinical evaluation
```

**Explanation**:
- **Training Set**: Model learns from these images
- **Validation Set**: Monitor model performance, tune hyperparameters
- **Test Set**: Final evaluation on unseen data

**Analogy**: 
- Train = Textbook examples
- Val = Practice exams
- Test = Final exam

---

### 7. **Meyerding Grading System**

**Definition**: Clinical classification system for spondylolisthesis severity.

**How it Works**: Measure how far one vertebra has slipped over another.

**Grades**:
```
Grade 0: No slip (Normal)           [====]
                                    [====]

Grade I: 0-25% slip                 [====]
                                     [====]

Grade II: 25-50% slip               [====]
                                       [====]

Grade III: 50-75% slip              [====]
                                         [====]

Grade IV: 75-100% slip              [====]
                                           [====]

Grade V: >100% (Spondyloptosis)     [====]
                                             [====]
```

**Calculation**: `Slip % = (displacement / vertebra width) × 100`

**Clinical Significance**:
- Grade 0-I: Usually managed conservatively
- Grade II-III: May need surgery
- Grade IV-V: Typically requires surgical intervention

---

### 8. **Metrics for Evaluation**

#### **MRE (Mean Radial Error)**
**Definition**: Average distance between predicted and actual keypoint locations.

**Formula**: `MRE = average(√[(x_pred - x_true)² + (y_pred - y_true)²])`

**What it Means**: 
- Lower is better
- Measured in pixels or millimeters
- Target: <2mm for clinical accuracy

**Example**:
```
True point:      (100, 150)
Predicted point: (102, 148)
Error = √[(102-100)² + (148-150)²] = √[4 + 4] = 2.83 pixels
```

---

#### **SDR (Success Detection Rate)**

**Definition**: Percentage of keypoints detected within a certain distance threshold.

**Your Metrics**:
- **SDR@2mm**: % of keypoints within 2mm of true location (strict)
- **SDR@4mm**: % of keypoints within 4mm of true location (lenient)

**Example**:
```
100 keypoints predicted
85 within 2mm → SDR@2mm = 85%
95 within 4mm → SDR@4mm = 95%
```

**Clinical Relevance**:
- 2mm threshold: Clinical precision required
- 4mm threshold: Acceptable for screening

---

#### **MAE (Mean Absolute Error)**

**Definition**: Average absolute difference between predicted and true values.

**For Slip Percentage**: How far off are your slip % predictions?

**Example**:
```
True slip:      32%
Predicted slip: 28%
MAE = |32 - 28| = 4%
```

**Target**: MAE < 3% for clinical grading accuracy

---

#### **Cohen's Kappa**

**Definition**: Statistical measure of agreement between two raters (or model vs. expert).

**Scale**:
```
0.0 - 0.2:  Slight agreement
0.2 - 0.4:  Fair agreement
0.4 - 0.6:  Moderate agreement
0.6 - 0.8:  Substantial agreement
0.8 - 1.0:  Almost perfect agreement
```

**Purpose**: Measure how well your model agrees with expert radiologists on Meyerding grades.

**Target**: κ > 0.8 (almost perfect agreement)

---

## Phase 1.1: Initial Data Exploration (Days 1-2)

### Objective
Understand the basic structure and statistics of your dataset.

---

### Task 1: Load and Parse Annotations

**What You'll Do**:
1. Navigate to your data folders
2. Load JSON files
3. Parse the structure

**Code Concept**:
```python
import json
import os
from pathlib import Path

# Path to your data
train_labels_path = "data/Train/Keypointrcnn_data/labels/train/"

# Load a sample annotation
with open(train_labels_path + "3209-F-073Y1_jpg.rf.xxx.json") as f:
    annotation = json.load(f)

# Understand structure
print("Boxes:", annotation['boxes'])       # 4 bounding boxes
print("Keypoints:", annotation['keypoints']) # 4 sets of 4 corners
print("Labels:", annotation['labels'])     # Class labels
```

**What to Look For**:
- Number of boxes = 4? ✓
- Number of keypoint sets = 4? ✓
- Each keypoint set has 4 points? ✓
- All labels are 0? ✓

---

### Task 2: Count Images and Annotations

**What You'll Do**:
Count total files in each split.

**Expected Output**:
```
Dataset Statistics:
------------------
Training images:   494
Validation images: 206
Test images:       16
Total images:      716

Training labels:   494
Validation labels: 206
Match: ✓
```

**Why Important**: Ensure every image has a corresponding annotation.

---

### Task 3: Analyze Filename Structure

**Your Filename Format**:
```
3209-F-073Y1_jpg.rf.b3686ffe92cdefd4f386611cd2115d2b.jpg
 ↑   ↑  ↑
 ID  Gender Age

Pattern: {PatientID}-{Gender}-{Age}Y1_jpg.rf.{Hash}.jpg
```

**Parse Information**:
- **Patient ID**: 3209
- **Gender**: F (Female) or M (Male)
- **Age**: 073 years → 73 years old

**What to Extract**:
1. Gender distribution (M vs F)
2. Age distribution (min, max, mean, median)
3. Patient demographics

**Expected Analysis**:
```
Gender Distribution:
- Male:   45%
- Female: 55%

Age Distribution:
- Min:    10 years
- Max:    87 years
- Mean:   54 years
- Median: 56 years
```

---

### Task 4: Vertebrae Coverage Analysis

**What to Check**:
Do all images have all 4 vertebrae annotated?

**Code Logic**:
```python
for each annotation:
    if len(boxes) == 4:
        complete_annotation += 1
    else:
        incomplete_annotation += 1
```

**Expected Output**:
```
Vertebrae Coverage:
- Complete (4 vertebrae): 710 images (99.2%)
- Incomplete:             6 images (0.8%)
```

---

### Deliverable 1.1: Summary Statistics Table

**Example Output**:

| Metric | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| Images | 494   | 206 | 16   | 716   |
| Male   | 220   | 95  | 8    | 323   |
| Female | 274   | 111 | 8    | 393   |
| Avg Age| 54.2  | 53.8| 56.1 | 54.1  |
| Complete Annotations | 490 | 204 | 16 | 710 |

---

## Phase 1.2: Visual Data Exploration (Days 3-4)

### Objective
**PRIMARY GOAL**: Visualize 10 random training images with keypoint annotations to understand what the data actually looks like.

---

### Why Visualization is Critical

**Reasons**:
1. **Verify annotations are correct**: Are keypoints actually on the vertebrae corners?
2. **Understand image quality**: Are X-rays clear? Any noise?
3. **Identify patterns**: How do vertebrae appear in different patients?
4. **Spot errors**: Missing keypoints, wrong labels, etc.
5. **Build intuition**: What makes detection challenging?

**Quote**: *"A picture is worth a thousand tables"* - In medical imaging, seeing the data is essential!

---

### Task 1: Plot 10 Random Images with Annotations

**Step-by-Step Process**:

#### Step 1: Select 10 Random Images
```python
import random

# Get all training image files
train_images = list(Path("data/Train/.../images/train/").glob("*.jpg"))

# Select 10 random images
random.seed(42)  # For reproducibility
sample_images = random.sample(train_images, 10)
```

#### Step 2: Load Image and Annotation
```python
import cv2
from PIL import Image

# For each sample
for img_path in sample_images:
    # Load image
    image = cv2.imread(str(img_path))
    
    # Find corresponding JSON
    json_path = img_path.name.replace('.jpg', '.json')
    
    # Load annotation
    with open(label_path / json_path) as f:
        annotation = json.load(f)
```

#### Step 3: Draw Keypoints and Boxes

**Vertebra Colors** (for clear distinction):
```python
colors = {
    0: (255, 0, 0),    # L3 = Red
    1: (0, 255, 0),    # L4 = Green
    2: (0, 0, 255),    # L5 = Blue
    3: (255, 255, 0)   # S1 = Yellow
}

vertebra_names = ['L3', 'L4', 'L5', 'S1']
```

**Drawing Logic**:
```python
for i, (box, keypoints) in enumerate(zip(annotation['boxes'], annotation['keypoints'])):
    color = colors[i]
    vertebra = vertebra_names[i]
    
    # Draw bounding box
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw keypoints (4 corners)
    for kp in keypoints:
        x, y, vis = kp
        if vis == 1:  # Only draw visible points
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
    
    # Label vertebra
    cv2.putText(image, vertebra, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

#### Step 4: Create Visualization Grid

**Layout**: 2 rows × 5 columns = 10 images

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(25, 10))

for idx, ax in enumerate(axes.flat):
    # Display annotated image
    ax.imshow(annotated_images[idx])
    ax.set_title(f"Sample {idx+1}: {patient_info[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('docs/figures/10_sample_images.png', dpi=150)
plt.show()
```

---

### What You Should See

**Expected Visualization**:
```
[Image 1]          [Image 2]          [Image 3]          [Image 4]          [Image 5]
3209-F-73Y         3210-M-67Y         3211-F-34Y         3213-M-57Y         3214-F-69Y

[Red L3]           [Red L3]           [Red L3]           [Red L3]           [Red L3]
[Green L4]         [Green L4]         [Green L4]         [Green L4]         [Green L4]
[Blue L5]          [Blue L5]          [Blue L5]          [Blue L5]          [Blue L5]
[Yellow S1]        [Yellow S1]        [Yellow S1]        [Yellow S1]        [Yellow S1]

[Image 6]          [Image 7]          [Image 8]          [Image 9]          [Image 10]
...
```

**Key Observations to Make**:
1. ✓ Are keypoints on vertebra corners?
2. ✓ Are bounding boxes tight around vertebrae?
3. ✓ Is there variation in vertebra size/shape?
4. ✓ Are some images clearer than others?
5. ✓ Do any annotations look wrong?

---

### Task 2: Draw Keypoint Connections

**Purpose**: Visualize the rectangular shape of each vertebra.

**Connection Pattern**:
```
Corner indices:
0 = Top-Left    1 = Top-Right
2 = Bottom-Left 3 = Bottom-Right

Connections:
0 ↔ 1  (top edge)
2 ↔ 3  (bottom edge)
0 ↔ 2  (left edge)
1 ↔ 3  (right edge)
```

**Code**:
```python
# Draw lines between corners
connections = [(0, 1), (2, 3), (0, 2), (1, 3)]

for start, end in connections:
    pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
    pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
    cv2.line(image, pt1, pt2, color, 2)
```

**Result**: Each vertebra will appear as a colored rectangle with corners marked.

---

### Task 3: Analyze Image Dimensions

**What to Check**:
- Are all images the same size?
- What's the typical resolution?
- Any outliers?

**Expected Findings**:
```
Image Dimension Analysis:
------------------------
Min size:    (480, 640)
Max size:    (1024, 768)
Most common: (512, 512) - 380 images (53%)
Aspect ratio: 1:1 to 4:3
```

**Why Important**: 
- Need to resize for neural networks
- Affects model input size
- May need padding/cropping strategy

---

### Task 4: Quality Assessment

**Visual Checklist**:
- [ ] X-ray contrast is good
- [ ] Vertebrae are clearly visible
- [ ] No major occlusions
- [ ] Annotations are accurate
- [ ] Keypoints align with anatomy

**Flag Issues**:
```
Quality Issues Found:
- 3 images with low contrast
- 1 image with missing vertebra annotation
- 2 images with potential misaligned keypoints
```

---

### Deliverable 1.2: Annotated Visualizations

**Files to Create**:
1. `docs/figures/10_random_samples.png` - Grid of 10 annotated images
2. `docs/figures/keypoint_connections.png` - Showing vertebra rectangles
3. `docs/figures/quality_examples.png` - Good vs. problematic images
4. `notebooks/00_data_understanding.ipynb` - Interactive notebook

**Documentation**:
```markdown
# Visual Exploration Summary

## Sample Images
- Plotted 10 random training images
- All vertebrae (L3, L4, L5, S1) clearly annotated
- Keypoints accurately placed on vertebra corners
- Color coding: L3(Red), L4(Green), L5(Blue), S1(Yellow)

## Observations
1. Image quality: Generally high, clear vertebrae visibility
2. Annotation accuracy: 98% correct placement
3. Vertebra detection: All 4 vertebrae present in 99% of images
4. Challenging cases: 3 images with low contrast identified

## Next Steps
- Proceed to statistical analysis
- Address 3 low-quality images in preprocessing
```

---

## Phase 1.3: Statistical Analysis (Days 5-6)

### Objective
Compute detailed statistics about keypoint distributions and clinical parameters.

---

### Task 1: Keypoint Distribution Analysis

#### Coordinate Ranges

**What to Calculate**:
```python
all_x_coords = []
all_y_coords = []

for annotation in all_annotations:
    for keypoint_set in annotation['keypoints']:
        for x, y, vis in keypoint_set:
            all_x_coords.append(x)
            all_y_coords.append(y)

print(f"X range: [{min(all_x_coords)}, {max(all_x_coords)}]")
print(f"Y range: [{min(all_y_coords)}, {max(all_y_coords)}]")
```

**Expected Output**:
```
Coordinate Ranges:
- X: [50, 1200] pixels
- Y: [0, 800] pixels

This tells us:
- Vertebrae span horizontally from x=50 to x=1200
- Vertically from top (y=0) to y=800
```

---

#### Keypoint Density Heatmap

**Purpose**: Visualize where keypoints are most concentrated.

**What it Shows**:
```
High density areas = Common vertebra locations
Low density areas  = Rare vertebra positions
```

**Interpretation**:
- Most vertebrae cluster in central spine region
- Helps understand anatomical variability
- Useful for region-based detection

---

#### Inter-Keypoint Distances

**What to Measure**: Distance between adjacent keypoints.

**Calculations**:
```python
# For each vertebra
top_width = distance(TL, TR)      # Top edge length
bottom_width = distance(BL, BR)   # Bottom edge length
left_height = distance(TL, BL)    # Left edge length
right_height = distance(TR, BR)   # Right edge length
```

**Expected Results**:
```
Vertebra Dimensions (in pixels):
                Min    Mean   Max    Std
Width (top):    45     68     95     12
Width (bottom): 40     65     92     13
Height (left):  30     48     70     10
Height (right): 28     46     68     11

Interpretation:
- Average vertebra is ~68 pixels wide
- Average vertebra is ~47 pixels tall
- Some vertebrae are larger (patients, X-ray zoom)
```

---

### Task 2: Compute Slip Percentages

**Clinical Context**: Spondylolisthesis severity is measured by how much one vertebra slips over another.

#### Step 1: Understand the Measurement

**Visual Explanation**:
```
Normal Alignment:
    [L4]
    [L5]

Slip Forward (Anterolisthesis):
    [L4]
     [L5]  ← L5 slipped forward

Measurement:
        |←---- w ---->|
        [    L4      ]
         [    L5    ]
        |← d →|

Slip % = (d / w) × 100
where:
  d = displacement (how far L5 moved)
  w = width of L4 (vertebra above)
```

#### Step 2: Calculate from Keypoints

**Algorithm**:
```python
def calculate_slip_percentage(L4_keypoints, L5_keypoints):
    # L4 bottom edge center
    L4_bottom_center_x = (L4_keypoints[2][0] + L4_keypoints[3][0]) / 2
    
    # L5 top edge center
    L5_top_center_x = (L5_keypoints[0][0] + L5_keypoints[1][0]) / 2
    
    # Displacement
    displacement = L5_top_center_x - L4_bottom_center_x
    
    # L4 width
    L4_width = L4_keypoints[3][0] - L4_keypoints[2][0]
    
    # Slip percentage
    slip_pct = (displacement / L4_width) * 100
    
    return slip_pct
```

#### Step 3: Compute Meyerding Grades

**Classification**:
```python
def meyerding_grade(slip_pct):
    if slip_pct < 0:
        return 0  # Retrolisthesis (backward slip)
    elif slip_pct < 25:
        return 1  # Grade I
    elif slip_pct < 50:
        return 2  # Grade II
    elif slip_pct < 75:
        return 3  # Grade III
    elif slip_pct < 100:
        return 4  # Grade IV
    else:
        return 5  # Grade V (Spondyloptosis)
```

#### Expected Distribution

**Example Output**:
```
Slip Percentage Distribution:
-----------------------------
Mean slip:     18.3%
Median slip:   12.5%
Std deviation: 15.2%

Meyerding Grade Distribution:
-----------------------------
Grade 0 (Normal):     320 images (44.7%)
Grade I (0-25%):      285 images (39.8%)
Grade II (25-50%):     85 images (11.9%)
Grade III (50-75%):    20 images (2.8%)
Grade IV (75-100%):     5 images (0.7%)
Grade V (>100%):        1 image  (0.1%)

Interpretation:
- Most patients have mild or no slip
- Only 3.6% have severe slips (Grade III+)
- Dataset is imbalanced (important for training!)
```

---

### Task 3: Bounding Box Analysis

**Calculations**:
```python
for box in all_boxes:
    x1, y1, x2, y2 = box
    
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = width / height
    
    box_widths.append(width)
    box_heights.append(height)
    box_areas.append(area)
    aspect_ratios.append(aspect_ratio)
```

**Expected Results**:
```
Bounding Box Statistics:
-----------------------
             Min    Mean   Max    Std
Width:       40     68     102    14
Height:      28     46     72     11
Area:        1200   3128   7344   1250
Aspect:      1.2    1.48   2.1    0.25

Interpretation:
- Vertebrae are wider than tall (aspect ~1.5)
- Consistent sizing (low std deviation)
- Predictable for model architecture design
```

---

### Task 4: Clinical Metadata Extraction

**Parse from Filenames**:
```python
import re

def parse_filename(filename):
    # Pattern: XXXX-G-YYYYY_jpg
    # Example: 3209-F-073Y1_jpg
    
    pattern = r'(\d+)-([MF])-(\d+)Y'
    match = re.search(pattern, filename)
    
    if match:
        patient_id = match.group(1)
        gender = 'Male' if match.group(2) == 'M' else 'Female'
        age = int(match.group(3))
        
        return patient_id, gender, age
    return None, None, None
```

**Correlation Analysis**:
```
Does slip severity correlate with age or gender?

Age vs. Slip:
- Correlation coefficient: 0.42 (moderate positive)
- Interpretation: Older patients tend to have higher slip %
- p-value: 0.001 (statistically significant)

Gender vs. Slip:
- Male mean slip:   16.8%
- Female mean slip: 19.2%
- p-value: 0.15 (not significant)
- Interpretation: No strong gender difference
```

---

### Deliverable 1.3: Statistical Reports

**Output Files**:
1. `docs/statistics/keypoint_stats.csv`
2. `docs/statistics/clinical_stats.csv`
3. `docs/figures/slip_distribution.png`
4. `docs/figures/meyerding_grades.png`
5. `docs/figures/correlation_plots.png`

**Summary Report**:
```markdown
# Statistical Analysis Summary

## Dataset Characteristics
- 716 total images
- 16 keypoints per image (4 vertebrae × 4 corners)
- Mean vertebra size: 68×47 pixels
- Coordinate range: X[50,1200], Y[0,800]

## Clinical Distribution
- 44.7% normal (Grade 0)
- 39.8% mild slip (Grade I)
- 15.5% moderate to severe (Grade II+)
- Class imbalance identified → need weighted sampling

## Key Findings
1. Strong age correlation with slip severity (r=0.42)
2. No significant gender difference
3. Consistent vertebra sizing (low variability)
4. Dataset suitable for deep learning (sufficient samples)
```

---

## Phase 1.4: Data Quality & Validation (Day 7)

### Objective
Final check to ensure data integrity before model training.

---

### Task 1: Validate Annotation Consistency

**Checks to Perform**:

#### Check 1: Image-Annotation Pairing
```python
image_files = set([f.stem for f in image_dir.glob('*.jpg')])
label_files = set([f.stem for f in label_dir.glob('*.json')])

missing_labels = image_files - label_files
missing_images = label_files - image_files

print(f"Images without labels: {len(missing_labels)}")
print(f"Labels without images: {len(missing_images)}")
```

**Expected**: Both should be 0.

---

#### Check 2: Keypoint Counts
```python
for json_file in all_annotations:
    with open(json_file) as f:
        data = json.load(f)
    
    n_boxes = len(data['boxes'])
    n_keypoint_sets = len(data['keypoints'])
    
    if n_boxes != 4 or n_keypoint_sets != 4:
        print(f"Warning: {json_file} has {n_boxes} boxes, {n_keypoint_sets} keypoint sets")
```

**Expected**: All should have exactly 4 boxes and 4 keypoint sets.

---

#### Check 3: Visibility Flags
```python
for annotation in all_annotations:
    for keypoint_set in annotation['keypoints']:
        for x, y, vis in keypoint_set:
            if vis not in [0, 1]:
                print(f"Invalid visibility flag: {vis}")
```

**Expected**: All visibility values should be 0 or 1.

---

#### Check 4: Coordinate Validity
```python
for annotation in all_annotations:
    image = load_image(annotation['image_path'])
    h, w = image.shape[:2]
    
    for keypoint_set in annotation['keypoints']:
        for x, y, vis in keypoint_set:
            if x < 0 or x >= w or y < 0 or y >= h:
                print(f"Keypoint out of bounds: ({x}, {y}) in {w}×{h} image")
```

**Expected**: All keypoints should be within image boundaries.

---

### Task 2: Identify Edge Cases

**Categories to Flag**:

#### 1. Low Contrast Images
```python
# Calculate image contrast
contrast = image.std()

if contrast < threshold:
    flag_as_low_contrast(image_path)
```

#### 2. Unusual Vertebra Positions
```python
# Check if vertebrae are in expected order
vertebra_y_centers = [compute_center_y(kps) for kps in keypoints]

# Should be monotonically increasing (L3 → L4 → L5 → S1)
if not is_monotonic(vertebra_y_centers):
    flag_as_unusual_anatomy(image_path)
```

#### 3. Extreme Slip Cases
```python
if slip_percentage > 75:  # Grade IV or V
    flag_as_extreme_case(image_path)
```

#### 4. Missing Vertebrae
```python
if len(annotation['boxes']) < 4:
    flag_as_incomplete(image_path)
```

---

### Task 3: Data Cleaning Recommendations

**Output Report**:
```
Data Quality Assessment Report
==============================

Total images analyzed: 716

✓ PASSED CHECKS (710 images):
  - All have matching image-label pairs
  - All have 4 vertebrae annotated
  - All keypoints within image bounds
  - All visibility flags valid

⚠ FLAGGED ISSUES (6 images):

Low Contrast (3 images):
  - 3704-M-045Y1_jpg.rf.xxx.jpg
  - 3705-F-071Y1_jpg.rf.xxx.jpg
  - 3709-M-007Y1_jpg.rf.xxx.jpg
  Recommendation: Apply histogram equalization

Unusual Anatomy (2 images):
  - N30-Olisthesis-L3-4-and-L4-5-F-48-Yrs.jpg (multi-level slip)
  - 3341-M-085Y1_jpg.rf.xxx.jpg (rotated vertebra)
  Recommendation: Include in training for robustness

Extreme Cases (1 image):
  - N48-Olisthesis-L3-4-L4-5-L5-S1-F-55-Yrs.jpg (Grade V)
  Recommendation: Keep for testing edge case performance

RECOMMENDATIONS:
1. Apply contrast enhancement to 3 images
2. Include all images in training (diversity is good)
3. No exclusions necessary
4. Proceed to model development
```

---

### Task 4: Preprocessing Requirements

**Determined Needs**:

1. **Image Resizing**:
   - Target size: 512×512 pixels
   - Method: Resize with padding to maintain aspect ratio

2. **Normalization**:
   - Pixel values: [0, 255] → [0, 1]
   - Mean/std: Compute from training set

3. **Keypoint Scaling**:
   - Scale coordinates proportionally with image resize
   - Important: Must maintain anatomical accuracy

4. **Data Augmentation** (for training):
   - Random rotation: ±10°
   - Random scaling: 0.9-1.1×
   - Random translation: ±10 pixels
   - Brightness/contrast adjustment
   - **Note**: Must transform keypoints accordingly!

---

### Deliverable 1.4: Quality Assurance Report

**Files**:
1. `docs/quality/data_validation_report.md`
2. `docs/quality/flagged_images.csv`
3. `docs/quality/preprocessing_requirements.md`

---

## Expected Outputs (End of Phase 1)

### 1. Jupyter Notebook
**File**: `notebooks/00_data_understanding.ipynb`

**Contents**:
- All Phase 1 code
- Interactive visualizations
- Statistical computations
- Quality checks
- Well-documented with markdown cells

---

### 2. Figures Directory
**Location**: `docs/figures/`

**Files**:
- `10_random_samples.png` - Grid of annotated images
- `keypoint_connections.png` - Vertebra rectangles
- `slip_distribution.png` - Histogram of slip %
- `meyerding_grades.png` - Bar chart of grades
- `age_distribution.png` - Patient age histogram
- `gender_distribution.png` - Male vs Female pie chart
- `vertebra_dimensions.png` - Box plots of sizes
- `correlation_matrix.png` - Age/gender vs slip

---

### 3. Statistics Files
**Location**: `docs/statistics/`

**Files**:
- `dataset_summary.csv` - Overall statistics
- `keypoint_statistics.csv` - Coordinate ranges, distances
- `clinical_parameters.csv` - Slip %, Meyerding grades
- `quality_metrics.csv` - Image quality scores

---

### 4. Documentation
**Location**: `docs/`

**Files**:
- `data_exploration_report.md` - Complete findings
- `annotation_format.md` - JSON structure explanation
- `preprocessing_guide.md` - Data preparation steps

---

## Troubleshooting Guide

### Issue 1: JSON Files Won't Load

**Symptom**: `json.decoder.JSONDecodeError`

**Solutions**:
```python
# Check file encoding
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Validate JSON structure
import jsonschema

# Or use try-except
try:
    data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error in {json_path}: {e}")
```

---

### Issue 2: Image Won't Display

**Symptom**: Black or corrupted images

**Solutions**:
```python
# Try different loading methods
image = cv2.imread(str(image_path))  # OpenCV
# or
image = Image.open(image_path)        # PIL
# or
image = plt.imread(str(image_path))   # Matplotlib

# Check if image loaded
if image is None:
    print(f"Failed to load: {image_path}")

# Check image format
print(f"Shape: {image.shape}")
print(f"Dtype: {image.dtype}")
print(f"Range: [{image.min()}, {image.max()}]")
```

---

### Issue 3: Keypoints Appear in Wrong Positions

**Symptom**: Points not on vertebra corners

**Possible Causes**:
1. **Image coordinate system confusion**:
   - OpenCV: (x, y) = (column, row)
   - Matplotlib: (x, y) = (column, row)
   - Some libraries use (row, col) instead!

2. **Solution**:
```python
# Always verify coordinate order
x, y, vis = keypoint
print(f"X (horizontal): {x}, Y (vertical): {y}")

# Test on a known image
cv2.circle(image, (int(x), int(y)), 5, (255,0,0), -1)
plt.imshow(image)
```

---

### Issue 4: Out of Memory Errors

**Symptom**: `MemoryError` or kernel crash

**Solutions**:
```python
# Don't load all images at once
# Instead, process in batches

# BAD:
all_images = [cv2.imread(p) for p in image_paths]  # ❌

# GOOD:
for img_path in image_paths:
    image = cv2.imread(str(img_path))
    process_image(image)
    del image  # Free memory
    
# Or use generators
def image_generator(paths):
    for path in paths:
        yield cv2.imread(str(path))
```

---

### Issue 5: Slow Processing

**Symptom**: Takes too long to process all images

**Solutions**:
```python
# Use progress bars
from tqdm import tqdm

for img_path in tqdm(image_paths, desc="Processing images"):
    process_image(img_path)

# Parallel processing
from multiprocessing import Pool

def process_wrapper(path):
    return process_image(path)

with Pool(4) as pool:  # 4 parallel workers
    results = pool.map(process_wrapper, image_paths)
```

---

## Summary Checklist

At the end of Phase 1, you should have:

- [x] **Understanding**:
  - ✓ Know annotation format (boxes, keypoints, labels)
  - ✓ Understand clinical context (Meyerding grading)
  - ✓ Grasp dataset statistics (716 images, 4 vertebrae, etc.)

- [x] **Visualizations**:
  - ✓ 10 random annotated images plotted
  - ✓ Keypoint connections drawn
  - ✓ Distribution plots created
  - ✓ All saved to `docs/figures/`

- [x] **Statistics**:
  - ✓ Computed slip percentages
  - ✓ Calculated Meyerding grades
  - ✓ Analyzed patient demographics
  - ✓ All saved to `docs/statistics/`

- [x] **Quality**:
  - ✓ Validated all annotations
  - ✓ Flagged problematic images (if any)
  - ✓ Documented preprocessing needs
  - ✓ Report created in `docs/quality/`

- [x] **Documentation**:
  - ✓ Comprehensive notebook with all code
  - ✓ Detailed exploration report
  - ✓ Ready to proceed to Phase 2

---

## Next Phase Preview

**Phase 2: Baseline U-Net Implementation**

Now that you understand your data thoroughly, you'll:
1. Design U-Net architecture for keypoint detection
2. Implement training pipeline
3. Evaluate first baseline performance
4. Compare with statistical expectations from Phase 1

**Why Phase 1 was essential**:
- You know what "good" predictions look like
- You understand the clinical context
- You've identified challenges (class imbalance, etc.)
- You can make informed architectural decisions

---

*This guide is your reference throughout Phase 1. Bookmark it and return whenever you need clarification on terminology or methodology!*

**Questions?** Document them in your notebook as you work through Phase 1.

**Good luck! 🚀**
