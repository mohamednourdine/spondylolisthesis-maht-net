# Pixel Spacing Calibration Guide

## Overview

For **publication-ready results**, proper pixel spacing calibration is essential for reporting physical measurements (mm) alongside pixel-based metrics.

## Quick Start

### Option 1: Use Default (Pixel-based metrics only)
```python
from evaluation.keypoint_evaluator import get_global_evaluator

# Reports metrics in pixels only
evaluator = get_global_evaluator()
```

### Option 2: Configure Calibration (Recommended for clinical work)
```python
from evaluation.keypoint_evaluator import KeypointEvaluator
from evaluation.calibration import PixelSpacingCalibrator

# Load calibration
calibrator = PixelSpacingCalibrator(
    config_file=Path('config/pixel_spacing_calibration.json')
)

# Create evaluator with dual reporting (pixels + mm)
evaluator = KeypointEvaluator(
    calibrator=calibrator,
    report_physical_metrics=True
)
```

## Calibration Methods

### Method 1: Known Imaging Equipment
If you know your imaging equipment specifications:
```json
{
  "default_pixel_spacing_mm": 0.167,
  "dataset_info": {
    "equipment": "Digital Radiography System",
    "detector_pixel_size": "0.143 mm",
    "magnification_factor": "1.17"
  }
}
```

### Method 2: Calibration Markers
If images contain rulers or calibration markers:
1. Measure known distance in image (pixels)
2. Calculate: `pixel_spacing = known_distance_mm / measured_pixels`

### Method 3: Anatomical Reference (Least Accurate)
Use known anatomical dimensions:
- Average lumbar vertebra height: ~28-32mm
- Measure in image and compute ratio

## Typical Values

| Modality | Pixel Spacing (mm/pixel) | Notes |
|----------|-------------------------|-------|
| Digital X-ray | 0.14 - 0.20 | Most common for spine imaging |
| CT Scan | 0.50 - 1.00 | Depends on reconstruction |
| MRI | 0.50 - 2.00 | Highly variable |
| Ultrasound | 0.05 - 0.15 | High resolution |

## For Publication

### Report Both Metrics
Always include both pixel and physical measurements:

**Example Results Section:**
```
The proposed method achieved a mean radial error of 2.3 ± 0.8 pixels 
(0.38 ± 0.13 mm with pixel spacing of 0.167 mm/pixel) and SDR@2mm of 
94.5% on the validation set.
```

### Document Calibration Method
In your methods section, clearly state:
1. How pixel spacing was determined
2. Whether it's constant or varies per image
3. Calibration accuracy/uncertainty if known

### Best Practices
- ✅ Always report pixel-based metrics (resolution-independent)
- ✅ Include physical metrics (mm) when calibration available
- ✅ State calibration method in paper
- ✅ Compare with literature using same units
- ⚠️ Don't assume pixel spacing without verification
- ⚠️ Be transparent about calibration uncertainty

## Configuration File Format

```json
{
  "default_pixel_spacing_mm": 0.167,
  "dataset_info": {
    "type": "spine_xray",
    "source": "manual_measurement",
    "calibration_method": "Known detector specifications",
    "notes": "Based on equipment documentation"
  },
  "images": {
    "patient001.jpg": {
      "spacing": 0.165,
      "source": "per_image_calibration",
      "notes": "Adjusted for magnification"
    }
  }
}
```

## Example Usage

```python
# Training with calibration
from evaluation.calibration import PixelSpacingCalibrator
from evaluation.keypoint_evaluator import KeypointEvaluator

# Setup calibration
calibrator = PixelSpacingCalibrator(
    default_spacing=0.167,  # mm/pixel
    config_file=Path('config/pixel_spacing_calibration.json')
)

# Create evaluator
evaluator = KeypointEvaluator(
    sdr_thresholds_px=[6.0, 12.0, 18.0, 24.0],
    calibrator=calibrator,
    report_physical_metrics=True
)

# Results will include both:
# - MRE_px: 2.3 pixels
# - MRE_mm: 0.38 mm
# - SDR_2px: 0.945
# - SDR_0.3mm: 0.945 (same as 2px with 0.167 spacing)
```

## References

- DICOM Standard PS3.3: Pixel Spacing attribute (0028,0030)
- Fitzpatrick et al. "Target Registration Error..." IEEE TMI 2001
- Challenge evaluation protocols (ISBI, MICCAI)
