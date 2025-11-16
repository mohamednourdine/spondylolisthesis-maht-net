"""
Visualization utilities for preprocessing and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict, Optional, Tuple
import torch


def draw_keypoints_on_image(
    image: np.ndarray,
    keypoints: List[List[List[float]]],
    boxes: Optional[List[List[float]]] = None,
    draw_connections: bool = True,
    vertebra_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Draw keypoints and boxes on image.
    
    Args:
        image: Image array [H, W, 3] or [H, W]
        keypoints: Keypoints [[[x,y,v], ...], ...]
        boxes: Optional bounding boxes [[x1,y1,x2,y2], ...]
        draw_connections: Whether to connect keypoints
        vertebra_names: Optional names for each vertebra
        
    Returns:
        Annotated image
    """
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Ensure 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Make a copy
    annotated = image.copy()
    
    # Color palette (BGR for OpenCV)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (0, 165, 255),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    # Draw for each vertebra
    for i, vertebra_kps in enumerate(keypoints):
        color = colors[i % len(colors)]
        # Generate label dynamically for any number of vertebrae
        if vertebra_names is not None and i < len(vertebra_names):
            name = vertebra_names[i]
        else:
            name = f'V{i+1}'
        
        # Draw bounding box if provided
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, name, (x1, max(15, y1-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw keypoints
        visible_kps = []
        for x, y, vis in vertebra_kps:
            x, y = int(x), int(y)
            if vis > 0:
                cv2.circle(annotated, (x, y), 5, color, -1)
                cv2.circle(annotated, (x, y), 5, (255, 255, 255), 1)
                visible_kps.append((x, y))
        
        # Draw connections (rectangle)
        if draw_connections and len(visible_kps) >= 4:
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]
            for idx1, idx2 in connections:
                if idx1 < len(visible_kps) and idx2 < len(visible_kps):
                    cv2.line(annotated, visible_kps[idx1], visible_kps[idx2], 
                            color, 2)
    
    return annotated


def visualize_preprocessing_pipeline(
    original_image: np.ndarray,
    processed_sample: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize before and after preprocessing.
    
    Args:
        original_image: Original image
        processed_sample: Preprocessed sample from preprocess_sample()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original
    ax1 = axes[0]
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Processed
    ax2 = axes[1]
    processed_img = processed_sample['image']
    if isinstance(processed_img, torch.Tensor):
        processed_img = processed_img.permute(1, 2, 0).numpy()
    
    # Draw keypoints on processed image
    annotated = draw_keypoints_on_image(
        processed_img,
        processed_sample['keypoints'],
        processed_sample['boxes']
    )
    
    ax2.imshow(annotated)
    ax2.set_title('Preprocessed (512×512 with keypoints)', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add transform info
    transform_info = processed_sample['transform_info']
    info_text = (
        f"Scale: {transform_info['scale']:.3f}\n"
        f"Offset: ({transform_info['x_offset']}, {transform_info['y_offset']})"
    )
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.show()


def visualize_augmentation_effects(
    image: np.ndarray,
    keypoints: List[List[List[float]]],
    augmentation: 'SpondylolisthesisAugmentation',
    n_samples: int = 6,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize multiple augmentation results.
    
    Args:
        image: Preprocessed image
        keypoints: Preprocessed keypoints
        augmentation: Augmentation pipeline
        n_samples: Number of augmented samples to show
        save_path: Optional path to save figure
    """
    from .preprocessing import KeypointTransformer
    
    kp_transformer = KeypointTransformer()
    flat_kps = kp_transformer.flatten_keypoints(keypoints)
    
    # Create grid
    rows = (n_samples + 2) // 3
    cols = min(3, n_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, ax in enumerate(axes[:n_samples]):
        # Apply augmentation
        augmented = augmentation(image, flat_kps)
        
        # Convert tensor to numpy if needed
        aug_img = augmented['image']
        if isinstance(aug_img, torch.Tensor):
            aug_img = aug_img.permute(1, 2, 0).numpy()
        
        # Unflatten keypoints
        aug_kps = kp_transformer.unflatten_keypoints(
            augmented['keypoints'],
            keypoints
        )
        
        # Draw keypoints
        annotated = draw_keypoints_on_image(aug_img, aug_kps)
        
        ax.imshow(annotated)
        ax.set_title(f'Augmentation #{i+1}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmentation visualization: {save_path}")
    
    plt.show()


def plot_batch(
    batch: Dict,
    max_images: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of samples.
    
    Args:
        batch: Batch dictionary from DataLoader
        max_images: Maximum number of images to show
        save_path: Optional path to save figure
    """
    images = batch['images']
    keypoints = batch['keypoints']
    boxes = batch['boxes']
    
    n = min(max_images, len(images))
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    axes = [axes] if n == 1 else axes
    
    for i in range(n):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        
        kps = keypoints[i]
        if isinstance(kps, torch.Tensor):
            kps = kps.cpu().numpy().tolist()
        
        bxs = boxes[i]
        if isinstance(bxs, torch.Tensor):
            bxs = bxs.cpu().numpy().tolist()
        
        annotated = draw_keypoints_on_image(img, kps, bxs)
        
        axes[i].imshow(annotated)
        axes[i].set_title(f"Sample {i+1}", fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


class PreprocessingVisualizer:
    """
    Utility class for visualizing preprocessing steps.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        pass
    
    def visualize_preprocessing_steps(
        self,
        image_path,
        annotation: Dict,
        preprocessor,
        save_path: Optional[str] = None
    ):
        """
        Visualize preprocessing steps: original -> resized -> CLAHE.
        
        Args:
            image_path: Path to image file
            annotation: Annotation dictionary
            preprocessor: ImagePreprocessor instance
            save_path: Optional path to save figure
        """
        from pathlib import Path
        
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        resized, transform_info = preprocessor.resize_with_padding(image)
        enhanced = preprocessor.apply_clahe(resized)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image)
        axes[0].set_title(f"Original\n{image.shape[1]}×{image.shape[0]}", fontsize=12, pad=10)
        axes[0].axis('off')
        
        axes[1].imshow(resized)
        axes[1].set_title(f"Resized + Padded\n{resized.shape[1]}×{resized.shape[0]}", fontsize=12, pad=10)
        axes[1].axis('off')
        
        axes[2].imshow(enhanced)
        axes[2].set_title(f"CLAHE Enhanced\n{enhanced.shape[1]}×{enhanced.shape[0]}", fontsize=12, pad=10)
        axes[2].axis('off')
        
        plt.suptitle("Preprocessing Pipeline Steps", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_augmentations(
        self,
        image_path,
        annotation: Dict,
        preprocessor,
        augmentor,
        num_augmentations: int = 6,
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple augmented versions of an image.
        
        Args:
            image_path: Path to image file
            annotation: Annotation dictionary
            preprocessor: ImagePreprocessor instance
            augmentor: SpondylolisthesisAugmentation instance
            num_augmentations: Number of augmented versions to generate
            save_path: Optional path to save figure
        """
        from src.data.preprocessing import preprocess_sample, KeypointTransformer
        
        # Preprocess image
        result = preprocess_sample(
            image_path, annotation, preprocessor
        )
        
        preprocessed_image = result['image']
        preprocessed_keypoints = result['keypoints']
        
        # Prepare for augmentation
        prep_img = (preprocessed_image * 255).astype(np.uint8)
        
        # Flatten keypoints
        kp_transformer = KeypointTransformer()
        flat_kps = kp_transformer.flatten_keypoints(preprocessed_keypoints)
        
        # Create keypoint labels
        keypoint_labels = []
        for i in range(len(preprocessed_keypoints)):
            keypoint_labels.extend([i] * 4)
        
        # Generate augmentations
        augmented_samples = []
        for _ in range(num_augmentations):
            augmented = augmentor(
                image=prep_img,
                keypoints=flat_kps,
                keypoint_labels=keypoint_labels
            )
            
            # Unflatten keypoints
            aug_kps = kp_transformer.unflatten_keypoints(
                augmented['keypoints'],
                preprocessed_keypoints
            )
            
            # Compute boxes from augmented keypoints
            aug_boxes = kp_transformer.compute_boxes_from_keypoints(aug_kps)
            
            augmented_samples.append({
                'image': augmented['image'],
                'keypoints': aug_kps,
                'boxes': aug_boxes
            })
        
        # Visualize
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, aug_sample in enumerate(augmented_samples):
            aug_img = aug_sample['image']
            if isinstance(aug_img, torch.Tensor):
                img_np = aug_img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = aug_img
            
            # Draw keypoints and boxes
            img_with_annot = draw_keypoints_on_image(
                img_np, 
                aug_sample['keypoints'], 
                aug_sample['boxes']
            )
            
            axes[idx].imshow(img_with_annot)
            axes[idx].set_title(f"Augmentation {idx+1}", fontsize=12, pad=10)
            axes[idx].axis('off')
        
        plt.suptitle("Data Augmentation Variations (boxes from keypoints)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_batch(
        self,
        dataloader,
        num_samples: int = 4,
        save_path: Optional[str] = None
    ):
        """
        Visualize a batch from DataLoader.
        
        Args:
            dataloader: PyTorch DataLoader
            num_samples: Number of samples to visualize
            save_path: Optional path to save figure
        """
        # Get one batch
        batch = next(iter(dataloader))
        
        # Visualize
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx in range(min(num_samples, len(batch['images']))):
            # Get image and annotations
            image = batch['images'][idx].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            
            boxes = batch['boxes'][idx].numpy()
            keypoints = batch['keypoints'][idx].numpy()
            
            # Convert keypoints to list format for drawing
            keypoints_list = keypoints.tolist()
            
            # Draw annotations
            img_with_annot = draw_keypoints_on_image(image, keypoints_list, boxes)
            
            axes[idx].imshow(img_with_annot)
            axes[idx].set_title(f"Batch Sample {idx+1}", fontsize=12, pad=10)
            axes[idx].axis('off')
        
        plt.suptitle("DataLoader Batch Visualization", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
