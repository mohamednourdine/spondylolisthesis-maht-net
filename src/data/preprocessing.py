"""
Core preprocessing functions for spondylolisthesis dataset.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(
        self, 
        target_size: Union[int, Tuple[int, int]] = 512,
        normalize: bool = True,
        pad_color: int = 0,
        apply_clahe: bool = False
    ):
        """
        Args:
            target_size: Output image size as int (square) or tuple (width, height)
            normalize: Whether to normalize to [0, 1]
            pad_color: Padding color value
            apply_clahe: Whether to apply CLAHE contrast enhancement
        """
        # Handle both int and tuple for target_size
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.normalize = normalize
        self.pad_color = pad_color
        self.apply_clahe_flag = apply_clahe
    
    def resize_with_padding(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image while maintaining aspect ratio using padding.
        
        Args:
            image: Input image [H, W, 3] or [H, W]
            
        Returns:
            resized_image: Processed image
            transform_info: Dict with scale and padding info for coordinate transformation
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale to fit
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(
            image, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.full(
                (target_h, target_w, 3), 
                self.pad_color, 
                dtype=image.dtype
            )
        else:
            padded = np.full(
                (target_h, target_w), 
                self.pad_color, 
                dtype=image.dtype
            )
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        transform_info = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'target_size': self.target_size
        }
        
        return padded, transform_info
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Ensure uint8 format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            # Convert to LAB, apply to L channel only
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def preprocess(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            processed_image: Preprocessed image
            transform_info: Transformation metadata
        """
        # Apply CLAHE if enabled
        if self.apply_clahe_flag:
            image = self.apply_clahe(image)
        
        # Resize with padding
        processed, transform_info = self.resize_with_padding(image)
        
        # Normalize if requested
        if self.normalize:
            if processed.dtype == np.uint8:
                processed = processed.astype(np.float32) / 255.0
        
        return processed, transform_info
    
    def compute_dataset_statistics(
        self, 
        image_paths: List[Path]
    ) -> Dict[str, float]:
        """
        Compute mean and std for dataset normalization.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dict with mean and std per channel
        """
        print(f"Computing dataset statistics from {len(image_paths)} images...")
        
        means = []
        stds = []
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        
        mean = np.array(means).mean(axis=0)
        std = np.array(stds).mean(axis=0)
        
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        print(f"Dataset statistics computed:")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        
        return stats


class KeypointTransformer:
    """Handles keypoint coordinate transformations."""
    
    @staticmethod
    def transform_keypoints(
        keypoints: List[List[List[float]]],
        transform_info: Dict
    ) -> List[List[List[float]]]:
        """
        Transform keypoint coordinates after image preprocessing.
        
        Args:
            keypoints: Original keypoints [[[x,y,v], ...], ...]
                      Format: list of vertebrae, each with 4 keypoints
            transform_info: Transform parameters from resize_with_padding
            
        Returns:
            Transformed keypoints in same format
        """
        scale = transform_info['scale']
        x_offset = transform_info['x_offset']
        y_offset = transform_info['y_offset']
        
        transformed = []
        for vertebra_kps in keypoints:
            vertebra_transformed = []
            for kp in vertebra_kps:
                x, y, vis = kp[0], kp[1], kp[2]
                new_x = x * scale + x_offset
                new_y = y * scale + y_offset
                vertebra_transformed.append([new_x, new_y, vis])
            transformed.append(vertebra_transformed)
        
        return transformed
    
    @staticmethod
    def transform_boxes(
        boxes: List[List[float]],
        transform_info: Dict
    ) -> List[List[float]]:
        """
        Transform bounding boxes.
        
        Args:
            boxes: Original boxes [[x1, y1, x2, y2], ...]
            transform_info: Transform parameters
            
        Returns:
            Transformed boxes
        """
        scale = transform_info['scale']
        x_offset = transform_info['x_offset']
        y_offset = transform_info['y_offset']
        
        transformed = []
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            new_x1 = x1 * scale + x_offset
            new_y1 = y1 * scale + y_offset
            new_x2 = x2 * scale + x_offset
            new_y2 = y2 * scale + y_offset
            transformed.append([new_x1, new_y1, new_x2, new_y2])
        
        return transformed
    
    @staticmethod
    def flatten_keypoints(keypoints: List[List[List[float]]]) -> List[List[float]]:
        """
        Flatten nested keypoint structure for augmentation libraries.
        
        Args:
            keypoints: [[[x,y,v], ...], ...] format
            
        Returns:
            [[x, y], ...] flattened format (visibility removed)
        """
        flattened = []
        for vertebra_kps in keypoints:
            for kp in vertebra_kps:
                flattened.append([kp[0], kp[1]])  # x, y only
        return flattened
    
    @staticmethod
    def unflatten_keypoints(
        flattened: List[List[float]], 
        original_keypoints: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """
        Unflatten keypoints back to nested structure with visibility.
        
        Args:
            flattened: [[x, y], ...] format
            original_keypoints: Original structure to extract visibility values
            
        Returns:
            [[[x,y,v], ...], ...] format
        """
        unflattened = []
        idx = 0
        
        for vertebra_idx, vertebra_kps in enumerate(original_keypoints):
            vertebra_unflattened = []
            for kp_idx, orig_kp in enumerate(vertebra_kps):
                x, y = flattened[idx]
                vis = orig_kp[2]  # Keep original visibility
                vertebra_unflattened.append([x, y, vis])
                idx += 1
            unflattened.append(vertebra_unflattened)
        
        return unflattened
    
    @staticmethod
    def compute_boxes_from_keypoints(
        keypoints: List[List[List[float]]]
    ) -> List[List[float]]:
        """
        Compute bounding boxes from keypoint coordinates.
        
        This should be used after augmentation to ensure boxes match
        the transformed keypoints.
        
        Args:
            keypoints: [[[x,y,v], ...], ...] format
            
        Returns:
            boxes: [[x1, y1, x2, y2], ...] format
        """
        boxes = []
        for vertebra_kps in keypoints:
            # Extract x and y coordinates
            xs = [kp[0] for kp in vertebra_kps]
            ys = [kp[1] for kp in vertebra_kps]
            
            # Compute bounding box
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            
            boxes.append([x_min, y_min, x_max, y_max])
        
        return boxes


def preprocess_sample(
    image_path: Path,
    annotation: Dict,
    preprocessor: ImagePreprocessor,
    load_image: bool = True
) -> Dict:
    """
    Preprocess a single training sample.
    
    Args:
        image_path: Path to image file
        annotation: Annotation dictionary with boxes, keypoints, labels
        preprocessor: ImagePreprocessor instance
        load_image: Whether to load and process image (False for annotation-only)
        
    Returns:
        Dict with processed image, keypoints, boxes, and metadata
    """
    if load_image:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        processed_image, transform_info = preprocessor.preprocess(image)
    else:
        processed_image = None
        # Create dummy transform info for annotation-only processing
        transform_info = {
            'scale': 1.0,
            'x_offset': 0,
            'y_offset': 0,
            'original_size': (512, 512),
            'resized_size': (512, 512),
            'target_size': (512, 512)
        }
    
    # Transform annotations
    kp_transformer = KeypointTransformer()
    transformed_keypoints = kp_transformer.transform_keypoints(
        annotation['keypoints'], 
        transform_info
    )
    transformed_boxes = kp_transformer.transform_boxes(
        annotation['boxes'],
        transform_info
    )
    
    result = {
        'keypoints': transformed_keypoints,
        'boxes': transformed_boxes,
        'labels': annotation.get('labels', [0] * len(transformed_boxes)),
        'transform_info': transform_info,
        'filename': annotation.get('filename', image_path.name)
    }
    
    if load_image:
        result['image'] = processed_image
    
    return result


def validate_preprocessed_sample(sample: Dict) -> bool:
    """
    Validate that preprocessed sample is correct.
    
    Args:
        sample: Preprocessed sample dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required keys
        required_keys = ['keypoints', 'boxes', 'labels']
        for key in required_keys:
            if key not in sample:
                print(f"Missing key: {key}")
                return False
        
        # Check keypoint structure
        keypoints = sample['keypoints']
        if not isinstance(keypoints, list):
            print("Keypoints must be a list")
            return False
        
        # Check each vertebra has 4 keypoints
        for i, vertebra_kps in enumerate(keypoints):
            if len(vertebra_kps) != 4:
                print(f"Vertebra {i} has {len(vertebra_kps)} keypoints, expected 4")
                return False
            
            # Check each keypoint has [x, y, vis]
            for j, kp in enumerate(vertebra_kps):
                if len(kp) != 3:
                    print(f"Keypoint {j} of vertebra {i} has {len(kp)} values, expected 3")
                    return False
        
        # Check boxes match keypoints count
        if len(sample['boxes']) != len(keypoints):
            print(f"Boxes count ({len(sample['boxes'])}) != keypoints count ({len(keypoints)})")
            return False
        
        # Check coordinate ranges if image is present
        if 'image' in sample:
            target_size = sample['transform_info']['target_size']
            for vertebra_kps in keypoints:
                for x, y, vis in vertebra_kps:
                    if x < 0 or x > target_size[0] or y < 0 or y > target_size[1]:
                        print(f"Keypoint ({x}, {y}) out of bounds for size {target_size}")
                        return False
        
        return True
    
    except Exception as e:
        print(f"Validation error: {e}")
        return False
