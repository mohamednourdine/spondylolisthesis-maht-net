# Step 1: Dataset Preparation for BUU-LSPINE

## Overview

This guide covers preparing the BUU-LSPINE dataset for MAHT-Net training. The main task is creating a DataLoader that reads the CSV annotation format.

---

## Dataset Structure

```
data/buu-lspine/
├── AP/                              # Anterior-Posterior view
│   ├── 0001-F-037Y0.jpg            # Image file
│   ├── 0001-F-037Y0.csv            # Annotation file
│   ├── 0003-F-013Y0.jpg
│   ├── 0003-F-013Y0.csv
│   └── ... (3,600 pairs)
│
├── LA/                              # Lateral view
│   ├── 0001-F-037Y1.jpg
│   ├── 0001-F-037Y1.csv
│   └── ... (3,600 pairs)
│
└── KIOM2022_dataset_report.xlsx    # Dataset metadata
```

---

## Annotation Format Analysis

### CSV Structure

Each CSV file contains vertebral edge coordinates:

```
x1,y1,x2,y2,label
876.2222,167.0618,1111.472,168.7665,0
865.9939,313.6671,1119.996,313.6671,0
870.8508,344.4474,1116.587,351.1707,0
...
```

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| x1 | float | X-coordinate of left corner |
| y1 | float | Y-coordinate of left corner |
| x2 | float | X-coordinate of right corner |
| y2 | float | Y-coordinate of right corner |
| label | int | Spondylolisthesis status (0=normal, 1=affected) |

### Keypoint Mapping

**AP View (10 rows → 20 keypoints)**:
```
Row 0: L1 top edge    → keypoints 0-1 (left corner, right corner)
Row 1: L1 bottom edge → keypoints 2-3
Row 2: L2 top edge    → keypoints 4-5
Row 3: L2 bottom edge → keypoints 6-7
Row 4: L3 top edge    → keypoints 8-9
Row 5: L3 bottom edge → keypoints 10-11
Row 6: L4 top edge    → keypoints 12-13
Row 7: L4 bottom edge → keypoints 14-15
Row 8: L5 top edge    → keypoints 16-17
Row 9: L5 bottom edge → keypoints 18-19
```

**LA View (11 rows → 22 keypoints)**:
- Same structure plus S1 reference edge

---

## Implementation Steps

### Step 1.1: Create BUU-LSPINE Dataset Class

Create file: `src/data/buu_lspine_dataset.py`

```python
"""
BUU-LSPINE Dataset Loader for MAHT-Net.
Reads CSV annotation format with vertebral corner coordinates.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict, List, Optional


class BUULSpineDataset(Dataset):
    """
    Dataset class for BUU-LSPINE vertebral corner detection.
    
    Args:
        root_dir: Path to buu-lspine folder
        view: 'AP' or 'LA'
        split: 'train', 'val', or 'test'
        image_size: Target image size (H, W)
        sigma: Gaussian sigma for heatmap generation
        transform: Optional image transforms
        split_ratio: Tuple of (train, val, test) ratios
        seed: Random seed for reproducible splits
    """
    
    # Number of keypoints per view
    NUM_KEYPOINTS = {'AP': 20, 'LA': 22}
    
    def __init__(
        self,
        root_dir: str,
        view: str = 'AP',
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        sigma: float = 4.0,
        transform: Optional[callable] = None,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.view = view.upper()
        self.split = split
        self.image_size = image_size
        self.sigma = sigma
        self.transform = transform
        
        assert self.view in ['AP', 'LA'], f"View must be 'AP' or 'LA', got {view}"
        
        self.num_keypoints = self.NUM_KEYPOINTS[self.view]
        self.data_dir = self.root_dir / self.view
        
        # Get all image files
        all_images = sorted(self.data_dir.glob('*.jpg'))
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        # Create train/val/test split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        
        n_train = int(len(all_images) * split_ratio[0])
        n_val = int(len(all_images) * split_ratio[1])
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        else:  # test
            self.indices = indices[n_train + n_val:]
        
        self.images = [all_images[i] for i in self.indices]
        
        print(f"Loaded {len(self.images)} {self.view} images for {split}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'image': (3, H, W) tensor
                - 'keypoints': (K, 2) tensor of (x, y) coordinates
                - 'heatmaps': (K, H, W) tensor of Gaussian heatmaps
                - 'spondy_labels': (num_edges,) tensor of spondylolisthesis labels
                - 'meta': dict with image_path, original_size, scale_factors
        """
        img_path = self.images[idx]
        csv_path = img_path.with_suffix('.csv')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize image
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Calculate scale factors
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        
        # Load annotations from CSV
        keypoints, spondy_labels = self._load_csv_annotations(csv_path)
        
        # Scale keypoints to new image size
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints)
        
        # Apply transforms if any
        if self.transform:
            # Transform expects dict with image, keypoints
            transformed = self.transform({
                'image': image,
                'keypoints': keypoints
            })
            image = transformed['image']
            keypoints = transformed['keypoints']
            # Regenerate heatmaps after transform
            heatmaps = self._generate_heatmaps(keypoints)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        keypoints = torch.from_numpy(keypoints).float()  # (K, 2)
        heatmaps = torch.from_numpy(heatmaps).float()  # (K, H, W)
        spondy_labels = torch.from_numpy(spondy_labels).long()
        
        return {
            'image': image,
            'keypoints': keypoints,
            'heatmaps': heatmaps,
            'spondy_labels': spondy_labels,
            'meta': {
                'image_path': str(img_path),
                'original_size': (orig_h, orig_w),
                'scale_factors': (scale_y, scale_x)
            }
        }
    
    def _load_csv_annotations(
        self, csv_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load keypoints from CSV file.
        
        Returns:
            keypoints: (K, 2) array of (x, y) coordinates
            spondy_labels: (num_edges,) array of labels
        """
        df = pd.read_csv(csv_path, header=None)
        
        # Expected columns: x1, y1, x2, y2, label
        num_edges = len(df)
        expected_edges = 10 if self.view == 'AP' else 11
        
        assert num_edges == expected_edges, \
            f"Expected {expected_edges} edges, got {num_edges} in {csv_path}"
        
        # Extract keypoints (2 corners per edge)
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        spondy_labels = np.zeros(num_edges, dtype=np.int64)
        
        for i, row in df.iterrows():
            x1, y1, x2, y2, label = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]
            
            # Left corner
            keypoints[i * 2, 0] = x1
            keypoints[i * 2, 1] = y1
            
            # Right corner
            keypoints[i * 2 + 1, 0] = x2
            keypoints[i * 2 + 1, 1] = y2
            
            spondy_labels[i] = int(label)
        
        return keypoints, spondy_labels
    
    def _generate_heatmaps(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmaps for all keypoints.
        
        Args:
            keypoints: (K, 2) array of (x, y) coordinates
            
        Returns:
            heatmaps: (K, H, W) array
        """
        H, W = self.image_size
        K = len(keypoints)
        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        
        for k in range(K):
            x, y = keypoints[k]
            
            # Skip invalid keypoints
            if x < 0 or y < 0 or x >= W or y >= H:
                continue
            
            # Generate Gaussian
            heatmaps[k] = self._gaussian_2d(
                center=(x, y),
                sigma=self.sigma,
                shape=(H, W)
            )
        
        return heatmaps
    
    def _gaussian_2d(
        self, 
        center: Tuple[float, float],
        sigma: float,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Generate 2D Gaussian centered at (cx, cy)."""
        H, W = shape
        cx, cy = center
        
        # Create coordinate grids
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        
        # Compute Gaussian
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        return gaussian.astype(np.float32)


def create_dataloaders(
    root_dir: str,
    view: str = 'AP',
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, val, test dataloaders for BUU-LSPINE.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = BUULSpineDataset(root_dir, view, 'train', image_size)
    val_dataset = BUULSpineDataset(root_dir, view, 'val', image_size)
    test_dataset = BUULSpineDataset(root_dir, view, 'test', image_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### Step 1.2: Create Test Script

Create file: `scripts/test_buu_lspine_dataset.py`

```python
#!/usr/bin/env python3
"""Test the BUU-LSPINE dataset loader."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.data.buu_lspine_dataset import BUULSpineDataset


def test_dataset():
    """Test loading and visualizing samples."""
    
    root_dir = "data/buu-lspine"
    
    # Test AP view
    print("Testing AP view...")
    ap_dataset = BUULSpineDataset(root_dir, view='AP', split='train')
    
    sample = ap_dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Keypoints shape: {sample['keypoints'].shape}")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Spondy labels: {sample['spondy_labels']}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image with keypoints
    img = sample['image'].permute(1, 2, 0).numpy()
    kps = sample['keypoints'].numpy()
    
    axes[0].imshow(img)
    axes[0].scatter(kps[:, 0], kps[:, 1], c='red', s=20)
    axes[0].set_title('Image with Keypoints')
    
    # Summed heatmaps
    heatmap_sum = sample['heatmaps'].sum(dim=0).numpy()
    axes[1].imshow(heatmap_sum, cmap='hot')
    axes[1].set_title('Summed Heatmaps')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(heatmap_sum, alpha=0.5, cmap='hot')
    axes[2].set_title('Overlay')
    
    plt.tight_layout()
    plt.savefig('experiments/visualizations/dataset_test_ap.png')
    print("Saved visualization to experiments/visualizations/dataset_test_ap.png")
    
    # Test LA view
    print("\nTesting LA view...")
    la_dataset = BUULSpineDataset(root_dir, view='LA', split='train')
    sample_la = la_dataset[0]
    print(f"LA Keypoints shape: {sample_la['keypoints'].shape}")
    
    print("\n✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
```

### Step 1.3: Update Configuration

Update `config/base_config.py`:

```python
# Change these paths:
DATA_ROOT = PROJECT_ROOT / 'data' / 'buu-lspine'
AP_DIR = DATA_ROOT / 'AP'
LA_DIR = DATA_ROOT / 'LA'

# Add new settings:
NUM_KEYPOINTS_AP = 20  # 10 edges × 2 corners
NUM_KEYPOINTS_LA = 22  # 11 edges × 2 corners
```

---

## Step 1.4: Data Augmentation

Create augmentation pipeline in `src/data/augmentation.py`:

```python
"""
Data augmentation for vertebral keypoint detection.
Augmentations must transform both image and keypoints consistently.
"""

import numpy as np
import cv2
from typing import Dict


class BUULSpineAugmentation:
    """
    Augmentation pipeline for BUU-LSPINE dataset.
    Carefully designed for medical imaging + keypoint detection.
    """
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        rotation_range: float = 10.0,  # degrees
        scale_range: tuple = (0.9, 1.1),
        brightness_range: tuple = (0.8, 1.2),
        contrast_range: tuple = (0.8, 1.2),
        elastic_alpha: float = 0,  # disabled by default (can distort anatomy)
        gaussian_noise_std: float = 0.02
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.elastic_alpha = elastic_alpha
        self.gaussian_noise_std = gaussian_noise_std
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply augmentations to image and keypoints.
        
        Args:
            sample: dict with 'image' (H, W, 3) and 'keypoints' (K, 2)
            
        Returns:
            Augmented sample
        """
        image = sample['image'].copy()
        keypoints = sample['keypoints'].copy()
        H, W = image.shape[:2]
        
        # 1. Horizontal flip (keypoints must be mirrored)
        if np.random.random() < self.horizontal_flip_prob:
            image = image[:, ::-1, :]
            keypoints[:, 0] = W - keypoints[:, 0]  # Mirror x-coordinates
            # Also swap left/right corner pairs
            keypoints = self._swap_left_right_corners(keypoints)
        
        # 2. Small rotation (preserve anatomy)
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        if abs(angle) > 0.1:
            image, keypoints = self._rotate(image, keypoints, angle)
        
        # 3. Scale
        scale = np.random.uniform(*self.scale_range)
        if abs(scale - 1.0) > 0.01:
            image, keypoints = self._scale(image, keypoints, scale)
        
        # 4. Brightness/Contrast (image only)
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        image = self._adjust_brightness_contrast(image, brightness, contrast)
        
        # 5. Gaussian noise (image only)
        if self.gaussian_noise_std > 0:
            noise = np.random.normal(0, self.gaussian_noise_std, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return {'image': image, 'keypoints': keypoints}
    
    def _swap_left_right_corners(self, keypoints: np.ndarray) -> np.ndarray:
        """Swap left and right corner pairs after horizontal flip."""
        K = len(keypoints)
        swapped = keypoints.copy()
        for i in range(0, K, 2):
            swapped[i], swapped[i+1] = keypoints[i+1].copy(), keypoints[i].copy()
        return swapped
    
    def _rotate(self, image, keypoints, angle):
        """Rotate image and keypoints around center."""
        H, W = image.shape[:2]
        cx, cy = W / 2, H / 2
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # Rotate image
        image = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        
        # Rotate keypoints
        ones = np.ones((len(keypoints), 1))
        kps_homo = np.hstack([keypoints, ones])
        keypoints = (M @ kps_homo.T).T
        
        return image, keypoints
    
    def _scale(self, image, keypoints, scale):
        """Scale image and keypoints."""
        H, W = image.shape[:2]
        new_H, new_W = int(H * scale), int(W * scale)
        
        image = cv2.resize(image, (new_W, new_H))
        keypoints = keypoints * scale
        
        # Crop or pad back to original size
        if scale > 1:
            # Crop center
            start_y = (new_H - H) // 2
            start_x = (new_W - W) // 2
            image = image[start_y:start_y+H, start_x:start_x+W]
            keypoints[:, 0] -= start_x
            keypoints[:, 1] -= start_y
        else:
            # Pad
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            padded = np.zeros((H, W, 3), dtype=image.dtype)
            padded[pad_y:pad_y+new_H, pad_x:pad_x+new_W] = image
            image = padded
            keypoints[:, 0] += pad_x
            keypoints[:, 1] += pad_y
        
        return image, keypoints
    
    def _adjust_brightness_contrast(self, image, brightness, contrast):
        """Adjust brightness and contrast."""
        image = image * contrast + (brightness - 1)
        return np.clip(image, 0, 1).astype(np.float32)
```

---

## Step 1.5: Pixel Spacing Calibration

The BUU-LSPINE annotations are in pixel coordinates. For clinical evaluation (MED in mm), we need pixel spacing calibration.

Check if calibration data exists:
```bash
ls data/buu-lspine/KIOM2022_dataset_report.xlsx
```

### Option A: Use Dataset Metadata

If the Excel file contains pixel spacing:
```python
import pandas as pd

df = pd.read_excel('data/buu-lspine/KIOM2022_dataset_report.xlsx')
# Look for columns like 'pixel_spacing', 'mm_per_pixel', etc.
print(df.columns)
```

### Option B: Estimate from Image Size

If no calibration available, estimate based on typical spine X-ray:
```python
# Typical lumbar spine X-ray: ~40cm coverage
# If image is 2000 pixels → 400mm / 2000 = 0.2 mm/pixel
DEFAULT_PIXEL_SPACING = 0.2  # mm/pixel (estimate)
```

### Option C: Per-Image Header

If DICOM metadata available in image headers, extract from there.

---

## Verification Checklist

Run these commands to verify dataset preparation:

```bash
# 1. Check file counts
echo "AP images:" && ls data/buu-lspine/AP/*.jpg | wc -l
echo "AP annotations:" && ls data/buu-lspine/AP/*.csv | wc -l
echo "LA images:" && ls data/buu-lspine/LA/*.jpg | wc -l
echo "LA annotations:" && ls data/buu-lspine/LA/*.csv | wc -l

# Expected: 3600 for each

# 2. Test dataset loader
python scripts/test_buu_lspine_dataset.py

# 3. Check image dimensions
python -c "
from PIL import Image
from pathlib import Path
imgs = list(Path('data/buu-lspine/AP').glob('*.jpg'))[:5]
for p in imgs:
    img = Image.open(p)
    print(f'{p.name}: {img.size}')
"
```

---

## Common Issues & Solutions

### Issue 1: CSV parsing errors

```python
# Some CSVs may have header or different separators
# Try:
df = pd.read_csv(csv_path, header=None, sep=',')
# If that fails:
df = pd.read_csv(csv_path, header=None, sep='\t')
```

### Issue 2: Memory issues with large images

```python
# Resize images on load to save memory
image = image.resize((512, 512), Image.BILINEAR)
```

### Issue 3: Keypoints outside image bounds

```python
# Clip keypoints to valid range
keypoints[:, 0] = np.clip(keypoints[:, 0], 0, W - 1)
keypoints[:, 1] = np.clip(keypoints[:, 1], 0, H - 1)
```

---

## Next Step

After completing dataset preparation, proceed to:
- [02_maht_net_implementation.md](02_maht_net_implementation.md) - Implement the full MAHT-Net architecture

---

*Last Updated: February 2025*
