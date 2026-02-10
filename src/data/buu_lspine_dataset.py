"""
BUU-LSPINE Dataset Loader for MAHT-Net.
Reads CSV annotation format with vertebral corner coordinates.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional


class BUULSpineDataset(Dataset):
    """
    Dataset class for BUU-LSPINE vertebral corner detection.
    
    The BUU-LSPINE dataset contains X-ray images with CSV annotations
    for vertebral corner points.
    
    CSV format: x1, y1, x2, y2, label
        - (x1, y1): left corner coordinates
        - (x2, y2): right corner coordinates
        - label: spondylolisthesis status (0=normal, 1=affected)
    
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
    NUM_EDGES = {'AP': 10, 'LA': 11}
    
    # Edge labels for interpretation
    EDGE_LABELS_AP = [
        'L1-top', 'L1-bot', 'L2-top', 'L2-bot', 'L3-top',
        'L3-bot', 'L4-top', 'L4-bot', 'L5-top', 'L5-bot'
    ]
    EDGE_LABELS_LA = EDGE_LABELS_AP + ['S1-top']
    
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
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test'"
        
        self.num_keypoints = self.NUM_KEYPOINTS[self.view]
        self.num_edges = self.NUM_EDGES[self.view]
        self.data_dir = self.root_dir / self.view
        
        # Load pixel spacing data if available
        self.pixel_spacing = self._load_pixel_spacing()
        
        # Get all image files
        all_images = sorted(list(self.data_dir.glob('*.jpg')))
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        # Create train/val/test split (patient-level, reproducible)
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
        
        print(f"BUU-LSPINE {self.view} view | {split}: {len(self.images)} images")
    
    def _load_pixel_spacing(self) -> Dict[str, float]:
        """Load per-image pixel spacing from CSV if available."""
        spacing_file = self.root_dir / 'pixel_spacing.csv'
        if spacing_file.exists():
            df = pd.read_csv(spacing_file)
            # Create mapping: filename -> spacing
            spacing_dict = {}
            for _, row in df.iterrows():
                # Handle filename format (may or may not have view suffix)
                base_name = str(row['filename']).split('.')[0]  # Remove extension if any
                spacing_dict[base_name] = float(row['pixel_spacing_mm'])
            return spacing_dict
        return {}
    
    def _get_pixel_spacing(self, img_path: Path) -> float:
        """Get pixel spacing for a specific image."""
        # Extract patient ID from filename (e.g., "0001-F-037Y0.jpg" -> "0001-F-037Y")
        filename = img_path.stem
        base_name = filename[:-1] if filename[-1].isdigit() else filename  # Remove view suffix
        
        if base_name in self.pixel_spacing:
            return self.pixel_spacing[base_name]
        elif filename in self.pixel_spacing:
            return self.pixel_spacing[filename]
        else:
            # Default pixel spacing (most common value in BUU-LSPINE)
            return 0.175
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.
        
        Returns:
            dict with keys:
                - 'image': (3, H, W) tensor, normalized
                - 'keypoints': (K, 2) tensor of (x, y) coordinates (scaled to image_size)
                - 'heatmaps': (K, H, W) tensor of Gaussian heatmaps
                - 'spondy_labels': (num_edges,) tensor of spondylolisthesis labels
                - 'pixel_spacing': scalar, mm per pixel
                - 'meta': dict with image_path, original_size, scale_factors
        """
        img_path = self.images[idx]
        csv_path = img_path.with_suffix('.csv')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize image
        image = image.resize(self.image_size[::-1], Image.BILINEAR)  # (W, H)
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Calculate scale factors
        scale_x = self.image_size[1] / orig_w
        scale_y = self.image_size[0] / orig_h
        
        # Load annotations from CSV
        keypoints, spondy_labels = self._load_csv_annotations(csv_path)
        
        # Scale keypoints to new image size
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] *= scale_x
        keypoints_scaled[:, 1] *= scale_y
        
        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints_scaled)
        
        # Apply transforms if any (must handle both image and keypoints)
        if self.transform:
            transformed = self.transform({
                'image': image,
                'keypoints': keypoints_scaled
            })
            image = transformed['image']
            keypoints_scaled = transformed['keypoints']
            # Regenerate heatmaps after transform
            heatmaps = self._generate_heatmaps(keypoints_scaled)
        
        # Normalize image (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Get pixel spacing
        pixel_spacing = self._get_pixel_spacing(img_path)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
        keypoints_tensor = torch.from_numpy(keypoints_scaled).float()  # (K, 2)
        heatmaps = torch.from_numpy(heatmaps).float()  # (K, H, W)
        spondy_labels = torch.from_numpy(spondy_labels).long()
        
        return {
            'image': image,
            'keypoints': keypoints_tensor,
            'heatmaps': heatmaps,
            'spondy_labels': spondy_labels,
            'pixel_spacing': torch.tensor(pixel_spacing).float(),
            'meta': {
                'image_path': str(img_path),
                'original_size': (orig_h, orig_w),
                'scale_factors': (scale_y, scale_x),
                'keypoints_original': keypoints  # Original coordinates
            }
        }
    
    def _load_csv_annotations(
        self, csv_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load keypoints from CSV file.
        
        CSV format: x1, y1, x2, y2, label (no header)
        
        Returns:
            keypoints: (K, 2) array of (x, y) coordinates
            spondy_labels: (num_edges,) array of labels
        """
        df = pd.read_csv(csv_path, header=None)
        
        # Expected columns: x1, y1, x2, y2, label
        num_edges = len(df)
        expected_edges = self.num_edges
        
        if num_edges != expected_edges:
            raise ValueError(
                f"Expected {expected_edges} edges, got {num_edges} in {csv_path}"
            )
        
        # Extract keypoints (2 corners per edge)
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        spondy_labels = np.zeros(num_edges, dtype=np.int64)
        
        for i in range(num_edges):
            row = df.iloc[i]
            x1, y1, x2, y2, label = row[0], row[1], row[2], row[3], row[4]
            
            # Left corner
            keypoints[i * 2, 0] = float(x1)
            keypoints[i * 2, 1] = float(y1)
            
            # Right corner
            keypoints[i * 2 + 1, 0] = float(x2)
            keypoints[i * 2 + 1, 1] = float(y2)
            
            spondy_labels[i] = int(label)
        
        return keypoints, spondy_labels
    
    def _generate_heatmaps(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmaps for all keypoints.
        
        Args:
            keypoints: (K, 2) array of (x, y) coordinates
            
        Returns:
            heatmaps: (K, H, W) array with Gaussian peaks
        """
        H, W = self.image_size
        K = len(keypoints)
        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        
        # Pre-compute Gaussian size (6*sigma should cover most of the distribution)
        size = int(6 * self.sigma + 1)
        if size % 2 == 0:
            size += 1
        
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
        """
        Generate 2D Gaussian centered at (cx, cy).
        
        Uses efficient vectorized computation.
        """
        H, W = shape
        cx, cy = center
        
        # Create coordinate grids
        x = np.arange(W, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        # Compute Gaussian
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        return gaussian.astype(np.float32)
    
    def get_edge_label(self, edge_idx: int) -> str:
        """Get human-readable label for an edge index."""
        labels = self.EDGE_LABELS_AP if self.view == 'AP' else self.EDGE_LABELS_LA
        return labels[edge_idx] if edge_idx < len(labels) else f"edge_{edge_idx}"


def create_dataloaders(
    root_dir: str,
    view: str = 'AP',
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    sigma: float = 4.0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders for BUU-LSPINE.
    
    Args:
        root_dir: Path to buu-lspine folder
        view: 'AP' or 'LA'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (H, W)
        sigma: Gaussian sigma for heatmaps
        seed: Random seed for reproducible splits
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = BUULSpineDataset(
        root_dir, view, 'train', image_size, sigma, seed=seed
    )
    val_dataset = BUULSpineDataset(
        root_dir, view, 'val', image_size, sigma, seed=seed
    )
    test_dataset = BUULSpineDataset(
        root_dir, view, 'test', image_size, sigma, seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    dataset = BUULSpineDataset("data/buu-lspine", view='AP', split='train')
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Keypoints shape: {sample['keypoints'].shape}")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Pixel spacing: {sample['pixel_spacing']}")
