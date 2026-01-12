"""
UNet-specific dataset for heatmap-based keypoint detection.
Generates Gaussian heatmaps from keypoint annotations.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

from .dataset import SpondylolisthesisDataset
from .preprocessing import ImagePreprocessor
from .augmentation import SpondylolisthesisAugmentation


def generate_gaussian_heatmap(
    height: int,
    width: int,
    keypoint: np.ndarray,
    sigma: float = 3.0,
    amplitude: float = 1000.0
) -> np.ndarray:
    """
    Generate a Gaussian heatmap for a single keypoint.
    
    Args:
        height: Heatmap height
        width: Heatmap width
        keypoint: Keypoint [x, y, visibility]
        sigma: Standard deviation of Gaussian
        amplitude: Peak amplitude of the Gaussian (default: 1000.0 from working project)
        
    Returns:
        Heatmap array [H, W]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    x, y, vis = keypoint
    
    # Skip invisible keypoints
    if vis == 0:
        return heatmap
    
    # Ensure keypoint is within bounds
    if x < 0 or x >= width or y < 0 or y >= height:
        return heatmap
    
    # Generate Gaussian
    x, y = int(x), int(y)
    
    # Create meshgrid
    tmp_size = sigma * 3
    ul = [int(x - tmp_size), int(y - tmp_size)]
    br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
    
    # Check bounds
    if ul[0] >= width or ul[1] >= height or br[0] < 0 or br[1] < 0:
        return heatmap
    
    # Generate gaussian
    size = 2 * tmp_size + 1
    x_range = np.arange(0, size, 1, np.float32)
    y_range = x_range[:, np.newaxis]
    x0 = y0 = size // 2
    
    # The gaussian peak is set to amplitude (not 1.0)
    g = amplitude * np.exp(-((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))
    
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
    
    # Image range
    img_x = max(0, ul[0]), min(br[0], width)
    img_y = max(0, ul[1]), min(br[1], height)
    
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    return heatmap


def generate_heatmaps_from_keypoints(
    keypoints: np.ndarray,
    image_height: int,
    image_width: int,
    max_vertebrae: int = 10,
    corners_per_vertebra: int = 4,
    sigma: float = 3.0,
    amplitude: float = 1000.0
) -> np.ndarray:
    """
    Generate per-vertebra heatmaps from keypoint annotations.
    
    Args:
        keypoints: Array of shape [N, 4, 3] where N is number of vertebrae,
                   4 is corners per vertebra, 3 is [x, y, visibility]
        image_height: Target heatmap height
        image_width: Target heatmap width
        max_vertebrae: Maximum number of vertebrae (for padding)
        corners_per_vertebra: Number of corners per vertebra (4)
        sigma: Gaussian sigma for heatmap generation
        amplitude: Peak amplitude (default: 1000.0 from working project)
        
    Returns:
        Heatmaps array [max_vertebrae * corners_per_vertebra, H, W]
        Each vertebra has 4 consecutive channels for its corners.
    """
    num_channels = max_vertebrae * corners_per_vertebra
    heatmaps = np.zeros((num_channels, image_height, image_width), dtype=np.float32)
    
    # Sort vertebrae by Y-coordinate (top to bottom)
    # Use center Y of each vertebra for sorting
    vertebra_centers = []
    for vert_idx, vertebra_kps in enumerate(keypoints):
        # Get center Y from visible keypoints
        visible_y = [kp[1] for kp in vertebra_kps if kp[2] > 0]
        if visible_y:
            center_y = np.mean(visible_y)
            vertebra_centers.append((center_y, vert_idx))
    
    # Sort by Y position (top to bottom)
    vertebra_centers.sort(key=lambda x: x[0])
    
    # Generate heatmaps for each sorted vertebra
    for sorted_idx, (_, orig_idx) in enumerate(vertebra_centers):
        if sorted_idx >= max_vertebrae:
            break  # Skip if more vertebrae than max
        
        vertebra_kps = keypoints[orig_idx]
        
        # Generate heatmap for each corner of this vertebra
        for corner_idx in range(corners_per_vertebra):
            keypoint = vertebra_kps[corner_idx]
            channel_idx = sorted_idx * corners_per_vertebra + corner_idx
            
            heatmap = generate_gaussian_heatmap(
                image_height, image_width, keypoint, sigma, amplitude
            )
            heatmaps[channel_idx] = heatmap
    
    return heatmaps


class UNetSpondylolisthesisDataset(SpondylolisthesisDataset):
    """
    Dataset for UNet training - generates heatmaps from keypoints.
    
    Inherits from SpondylolisthesisDataset but returns heatmaps instead of
    raw keypoint coordinates.
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        mode: str = 'train',
        preprocessor: Optional[ImagePreprocessor] = None,
        augmentation: Optional[SpondylolisthesisAugmentation] = None,
        heatmap_sigma: float = 3.0,
        heatmap_amplitude: float = 10.0,
        max_vertebrae: int = 10,
        corners_per_vertebra: int = 4,
        output_stride: int = 1,
        target_size: Tuple[int, int] = (512, 512),
        apply_clahe: bool = True,
        **kwargs
    ):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing JSON annotations
            mode: 'train', 'val', or 'test'
            preprocessor: ImagePreprocessor instance
            augmentation: SpondylolisthesisAugmentation instance
            heatmap_sigma: Sigma for Gaussian heatmap generation
            heatmap_amplitude: Amplitude for Gaussian heatmap generation (default: 10.0)
            max_vertebrae: Maximum number of vertebrae per image (default: 10)
            corners_per_vertebra: Number of corners per vertebra (default: 4)
            output_stride: Downsampling factor for heatmaps (1 = same size as image)
            target_size: Image size (width, height) - default 512x512
            apply_clahe: Whether to apply CLAHE contrast enhancement
            **kwargs: Additional arguments for parent class
        """
        # Create preprocessor with custom size if not provided
        if preprocessor is None:
            preprocessor = ImagePreprocessor(
                target_size=target_size,
                normalize=True,
                apply_clahe=apply_clahe and (mode == 'train')
            )
        
        super().__init__(
            image_dir=image_dir,
            label_dir=label_dir,
            mode=mode,
            preprocessor=preprocessor,
            augmentation=augmentation,
            **kwargs
        )
        
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_amplitude = heatmap_amplitude
        self.max_vertebrae = max_vertebrae
        self.corners_per_vertebra = corners_per_vertebra
        self.output_stride = output_stride
        self.target_size = target_size
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample with heatmaps.
        
        Returns:
            Dict with:
                - image: Tensor [C, H, W]
                - heatmaps: Tensor [num_keypoint_types, H/output_stride, W/output_stride]
                - keypoints: Original keypoints for reference
                - filename: str
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)
        
        image = sample['image']
        keypoints = sample['keypoints'].numpy()
        
        # Get image dimensions
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        # Calculate heatmap dimensions
        heatmap_h = h // self.output_stride
        heatmap_w = w // self.output_stride
        
        # Scale keypoints if using output stride
        if self.output_stride > 1:
            scaled_keypoints = keypoints.copy()
            scaled_keypoints[:, :, 0] /= self.output_stride  # x
            scaled_keypoints[:, :, 1] /= self.output_stride  # y
        else:
            scaled_keypoints = keypoints
        
        # Generate heatmaps
        heatmaps = generate_heatmaps_from_keypoints(
            scaled_keypoints,
            heatmap_h,
            heatmap_w,
            self.max_vertebrae,
            self.corners_per_vertebra,
            self.heatmap_sigma,
            self.heatmap_amplitude
        )
        
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        
        return {
            'image': image,
            'heatmaps': heatmaps_tensor,
            'keypoints': sample['keypoints'],  # Keep original for reference
            'boxes': sample['boxes'],
            'labels': sample['labels'],
            'filename': sample['filename']
        }


def unet_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for UNet DataLoader.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    heatmaps = torch.stack([item['heatmaps'] for item in batch])
    
    # Keep keypoints, boxes, labels as lists (variable length)
    keypoints = [item['keypoints'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,
        'heatmaps': heatmaps,
        'keypoints': keypoints,
        'boxes': boxes,
        'labels': labels,
        'filenames': filenames
    }


def create_unet_dataloaders(
    train_image_dir: Path,
    train_label_dir: Path,
    val_image_dir: Path,
    val_label_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    heatmap_sigma: float = 3.0,
    output_stride: int = 1,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders for UNet.
    
    Args:
        train_image_dir: Training images directory
        train_label_dir: Training labels directory
        val_image_dir: Validation images directory
        val_label_dir: Validation labels directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        heatmap_sigma: Sigma for Gaussian heatmap generation
        output_stride: Downsampling factor for heatmaps
        **kwargs: Additional arguments for Dataset
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = UNetSpondylolisthesisDataset(
        train_image_dir,
        train_label_dir,
        mode='train',
        heatmap_sigma=heatmap_sigma,
        output_stride=output_stride,
        **kwargs
    )
    
    val_dataset = UNetSpondylolisthesisDataset(
        val_image_dir,
        val_label_dir,
        mode='val',
        heatmap_sigma=heatmap_sigma,
        output_stride=output_stride,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=unet_collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=unet_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
