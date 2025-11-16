"""
PyTorch Dataset classes for Spondylolisthesis MAHT-Net.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from .preprocessing import ImagePreprocessor, KeypointTransformer, preprocess_sample
from .augmentation import SpondylolisthesisAugmentation


class SpondylolisthesisDataset(Dataset):
    """
    PyTorch Dataset for Spondylolisthesis detection with keypoints.
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        mode: str = 'train',
        preprocessor: Optional[ImagePreprocessor] = None,
        augmentation: Optional[SpondylolisthesisAugmentation] = None,
        cache_preprocessed: bool = False
    ):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing JSON annotations
            mode: 'train', 'val', or 'test'
            preprocessor: ImagePreprocessor instance
            augmentation: SpondylolisthesisAugmentation instance
            cache_preprocessed: Whether to cache preprocessed samples in memory
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        
        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = ImagePreprocessor(
                target_size=(512, 512),
                normalize=True,
                apply_clahe=(mode == 'train')
            )
        else:
            self.preprocessor = preprocessor
        
        # Initialize augmentation
        if augmentation is None and mode == 'train':
            self.augmentation = SpondylolisthesisAugmentation(mode='train')
        else:
            self.augmentation = augmentation
        
        self.cache_preprocessed = cache_preprocessed
        self.cache = {}
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        print(f"Loaded {len(self.annotations)} samples for {mode} mode")
    
    def _load_annotations(self) -> List[Dict]:
        """Load all JSON annotations."""
        annotations = []
        json_files = sorted(list(self.label_dir.glob('*.json')))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Find corresponding image
                img_name = json_file.stem
                img_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = self.image_dir / f"{img_name}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path is None:
                    print(f"Warning: No image found for {json_file.name}")
                    continue
                
                data['image_path'] = str(img_path)
                data['filename'] = json_file.name
                annotations.append(data)
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return annotations
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dict with:
                - image: Tensor [C, H, W]
                - keypoints: Tensor [N, 4, 3] (N vertebrae, 4 corners, [x,y,vis])
                - boxes: Tensor [N, 4] (N vertebrae, [x1,y1,x2,y2])
                - labels: Tensor [N] (class labels)
                - filename: str
        """
        # Check cache first
        if self.cache_preprocessed and idx in self.cache:
            sample = self.cache[idx]
        else:
            # Load and preprocess
            annotation = self.annotations[idx]
            image_path = Path(annotation['image_path'])
            
            # Preprocess
            sample = preprocess_sample(
                image_path,
                annotation,
                self.preprocessor,
                load_image=True
            )
            
            # Cache if enabled
            if self.cache_preprocessed:
                self.cache[idx] = sample
        
        # Get image and keypoints
        image = sample['image']
        keypoints = sample['keypoints']
        boxes = sample['boxes']
        labels = sample['labels']
        
        # Apply augmentation if training
        if self.augmentation is not None and self.mode == 'train':
            # Flatten keypoints for augmentation
            kp_transformer = KeypointTransformer()
            flat_kps = kp_transformer.flatten_keypoints(keypoints)
            
            # Apply augmentation
            augmented = self.augmentation(image, flat_kps)
            
            # Unflatten keypoints
            aug_flat_kps = augmented['keypoints']
            keypoints = kp_transformer.unflatten_keypoints(aug_flat_kps, keypoints)
            image = augmented['image']
            
            # IMPORTANT: Recompute boxes from augmented keypoints
            # This ensures boxes match the transformed keypoint positions
            boxes = kp_transformer.compute_boxes_from_keypoints(keypoints)
        else:
            # Convert to tensor without augmentation
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert to tensors
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'image': image,
            'keypoints': keypoints_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'filename': sample['filename']
        }
    
    def get_sample_raw(self, idx: int) -> Dict:
        """Get a sample without augmentation (for visualization)."""
        annotation = self.annotations[idx]
        image_path = Path(annotation['image_path'])
        
        sample = preprocess_sample(
            image_path,
            annotation,
            self.preprocessor,
            load_image=True
        )
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Handles variable number of vertebrae per image.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Keep keypoints, boxes, labels as lists (variable length)
    keypoints = [item['keypoints'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,
        'keypoints': keypoints,
        'boxes': boxes,
        'labels': labels,
        'filenames': filenames
    }


def create_dataloaders(
    train_image_dir: Path,
    train_label_dir: Path,
    val_image_dir: Path,
    val_label_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_image_dir: Training images directory
        train_label_dir: Training labels directory
        val_image_dir: Validation images directory
        val_label_dir: Validation labels directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for Dataset
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SpondylolisthesisDataset(
        train_image_dir,
        train_label_dir,
        mode='train',
        **kwargs
    )
    
    val_dataset = SpondylolisthesisDataset(
        val_image_dir,
        val_label_dir,
        mode='val',
        **kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
