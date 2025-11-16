"""
Data augmentation pipeline for training.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SpondylolisthesisAugmentation:
    """Augmentation pipeline with keypoint-aware transforms."""
    
    def __init__(
        self, 
        mode: str = 'train',
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            mode: 'train', 'val', or 'test'
            image_size: Image dimensions (width, height)
        """
        self.mode = mode
        self.image_size = image_size
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build albumentations pipeline."""
        
        if self.mode == 'train':
            return A.Compose([
                # Geometric transforms
                A.ShiftScaleRotate(
                    shift_limit=0.05,       # ±5% translation
                    scale_limit=0.1,        # ±10% scaling
                    rotate_limit=10,        # ±10° rotation
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ),
                
                # Horizontal flip (vertical anatomy flip)
                A.HorizontalFlip(p=0.5),
                
                # Intensity transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                
                # Add noise
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
                ], p=0.3),
                
                # Blur/sharpening
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                ], p=0.3),
                
                # Simulated artifacts
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.1),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                ], p=0.2),
                
                # Ensure proper range
                A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=1.0),
                
                # Convert to tensor
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False,
                label_fields=['keypoint_labels']
            ))
        
        elif self.mode == 'val' or self.mode == 'test':
            return A.Compose([
                # Only normalization for validation/test
                A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=1.0),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False,
                label_fields=['keypoint_labels']
            ))
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train', 'val', or 'test'")
    
    def __call__(
        self, 
        image: np.ndarray, 
        keypoints: List[List[float]],
        keypoint_labels: Optional[List[int]] = None
    ) -> Dict:
        """
        Apply augmentation.
        
        Args:
            image: Preprocessed image [H, W, 3] or [H, W]
            keypoints: Flattened keypoints [[x, y], ...]
            keypoint_labels: Optional labels for each keypoint
            
        Returns:
            Dict with augmented image and keypoints
        """
        # Ensure image is in correct format
        if image.dtype == np.float64:
            image = image.astype(np.float32)
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Create default labels if not provided
        if keypoint_labels is None:
            keypoint_labels = list(range(len(keypoints)))
        
        # Apply transform
        transformed = self.transform(
            image=image,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels
        )
        
        return {
            'image': transformed['image'],
            'keypoints': transformed['keypoints'],
            'keypoint_labels': transformed.get('keypoint_labels', keypoint_labels)
        }
    
    def visualize_augmentation(
        self,
        image: np.ndarray,
        keypoints: List[List[float]],
        n_samples: int = 5
    ) -> List[Dict]:
        """
        Generate multiple augmented versions for visualization.
        
        Args:
            image: Input image
            keypoints: Input keypoints
            n_samples: Number of augmented samples to generate
            
        Returns:
            List of augmented samples
        """
        augmented_samples = []
        
        for _ in range(n_samples):
            augmented = self(image, keypoints)
            augmented_samples.append(augmented)
        
        return augmented_samples


class MixedPrecisionAugmentation:
    """
    Advanced augmentation with mixed precision and custom policies.
    """
    
    def __init__(
        self,
        mode: str = 'train',
        strong_aug_prob: float = 0.5
    ):
        """
        Args:
            mode: 'train' or 'val'
            strong_aug_prob: Probability of applying strong augmentations
        """
        self.mode = mode
        self.strong_aug_prob = strong_aug_prob
        self.weak_transform = self._build_weak_transform()
        self.strong_transform = self._build_strong_transform()
    
    def _build_weak_transform(self):
        """Build weak augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=1.0),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=['keypoint_labels']
        ))
    
    def _build_strong_transform(self):
        """Build strong augmentation pipeline."""
        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 70)),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2)),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.Sharpen(alpha=(0.3, 0.7)),
            ], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=1.0),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=['keypoint_labels']
        ))
    
    def __call__(
        self,
        image: np.ndarray,
        keypoints: List[List[float]],
        keypoint_labels: Optional[List[int]] = None
    ) -> Dict:
        """Apply weak or strong augmentation randomly."""
        
        if keypoint_labels is None:
            keypoint_labels = list(range(len(keypoints)))
        
        # Ensure proper format
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Choose transform
        if self.mode == 'train' and np.random.rand() < self.strong_aug_prob:
            transform = self.strong_transform
        else:
            transform = self.weak_transform
        
        return transform(
            image=image,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels
        )


def get_augmentation_pipeline(
    mode: str = 'train',
    aug_type: str = 'standard',
    **kwargs
) -> SpondylolisthesisAugmentation:
    """
    Factory function to get augmentation pipeline.
    
    Args:
        mode: 'train', 'val', or 'test'
        aug_type: 'standard' or 'mixed'
        **kwargs: Additional arguments for augmentation
        
    Returns:
        Augmentation pipeline instance
    """
    if aug_type == 'standard':
        return SpondylolisthesisAugmentation(mode=mode, **kwargs)
    elif aug_type == 'mixed':
        return MixedPrecisionAugmentation(mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
