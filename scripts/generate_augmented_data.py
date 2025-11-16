#!/usr/bin/env python3
"""
Script to pre-generate augmented training data.

This saves augmented images and labels to disk for cases where:
- You want to inspect all augmentations
- You need deterministic augmented datasets
- You want to share augmented data with others

Note: For training, on-the-fly augmentation is typically preferred.
"""

import sys
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import ImagePreprocessor, KeypointTransformer
from src.data.augmentation import SpondylolisthesisAugmentation


def generate_augmented_data(
    input_image_dir: Path,
    input_label_dir: Path,
    output_dir: Path,
    num_augmentations_per_image: int = 5,
    target_size: int = 512
):
    """
    Generate augmented versions of training data.
    
    Args:
        input_image_dir: Directory containing original images
        input_label_dir: Directory containing original label JSON files
        output_dir: Directory to save augmented data
        num_augmentations_per_image: Number of augmented versions per original image
        target_size: Target image size for preprocessing
    """
    # Create output directories
    output_image_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor and augmentor
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=False)
    augmentor = SpondylolisthesisAugmentation(mode='train')
    transformer = KeypointTransformer()
    
    # Get all image files
    image_files = sorted(list(input_image_dir.glob('*.jpg')) + list(input_image_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} original images")
    print(f"Generating {num_augmentations_per_image} augmentations per image")
    print(f"Total images to generate: {len(image_files) * (num_augmentations_per_image + 1)}")
    print(f"Output directory: {output_dir}")
    
    total_generated = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load annotation
        label_path = input_label_dir / f"{img_path.stem}.json"
        if not label_path.exists():
            print(f"Warning: No label found for {img_path.name}, skipping...")
            continue
        
        with open(label_path, 'r') as f:
            annotation = json.load(f)
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        resized_image, transform_info = preprocessor.resize_with_padding(image)
        
        # Transform keypoints and boxes
        keypoints = transformer.transform_keypoints(annotation['keypoints'], transform_info)
        boxes = transformer.transform_boxes(annotation['boxes'], transform_info)
        labels = annotation['labels']
        
        # Save original preprocessed version
        original_output_name = f"{img_path.stem}_orig"
        save_sample(
            resized_image, keypoints, boxes, labels,
            output_image_dir, output_label_dir, original_output_name
        )
        total_generated += 1
        
        # Generate augmented versions
        for aug_idx in range(num_augmentations_per_image):
            # Flatten keypoints for augmentation
            flat_kps = transformer.flatten_keypoints(keypoints)
            
            # Create keypoint labels (vertebra ID for each corner)
            keypoint_labels = []
            for i in range(len(keypoints)):
                keypoint_labels.extend([i] * 4)  # 4 corners per vertebra
            
            # Apply augmentation
            try:
                augmented = augmentor(
                    image=resized_image,
                    keypoints=flat_kps,
                    keypoint_labels=keypoint_labels
                )
                
                # Unflatten keypoints
                aug_keypoints = transformer.unflatten_keypoints(
                    augmented['keypoints'], keypoints
                )
                
                # Recompute boxes from augmented keypoints
                aug_boxes = transformer.compute_boxes_from_keypoints(aug_keypoints)
                
                # Convert tensor to numpy if needed
                aug_image = augmented['image']
                if hasattr(aug_image, 'permute'):  # It's a tensor
                    aug_image = aug_image.permute(1, 2, 0).numpy()
                    aug_image = (aug_image * 255).astype(np.uint8)
                
                # Save augmented version
                aug_output_name = f"{img_path.stem}_aug{aug_idx:02d}"
                save_sample(
                    aug_image, aug_keypoints, aug_boxes, labels,
                    output_image_dir, output_label_dir, aug_output_name
                )
                total_generated += 1
                
            except Exception as e:
                print(f"Error augmenting {img_path.name} (aug {aug_idx}): {e}")
                continue
    
    print(f"\n✅ Generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"Images saved to: {output_image_dir}")
    print(f"Labels saved to: {output_label_dir}")
    
    # Generate summary statistics
    summary = {
        'original_images': len(image_files),
        'augmentations_per_image': num_augmentations_per_image,
        'total_generated': total_generated,
        'output_directory': str(output_dir)
    }
    
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


def save_sample(image, keypoints, boxes, labels, image_dir, label_dir, filename):
    """Save a single sample (image + labels)."""
    # Save image
    image_path = image_dir / f"{filename}.jpg"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), image_bgr)
    
    # Save labels
    label_path = label_dir / f"{filename}.json"
    label_data = {
        'keypoints': keypoints,
        'boxes': boxes,
        'labels': labels
    }
    with open(label_path, 'w') as f:
        json.dump(label_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate augmented training data for spondylolisthesis detection'
    )
    parser.add_argument(
        '--input-images',
        type=str,
        required=True,
        help='Directory containing original training images'
    )
    parser.add_argument(
        '--input-labels',
        type=str,
        required=True,
        help='Directory containing original label JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for augmented data'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=5,
        help='Number of augmented versions per original image (default: 5)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=512,
        help='Target image size (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_image_dir = Path(args.input_images)
    input_label_dir = Path(args.input_labels)
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    if not input_image_dir.exists():
        raise ValueError(f"Input image directory does not exist: {input_image_dir}")
    if not input_label_dir.exists():
        raise ValueError(f"Input label directory does not exist: {input_label_dir}")
    
    # Generate augmented data
    generate_augmented_data(
        input_image_dir=input_image_dir,
        input_label_dir=input_label_dir,
        output_dir=output_dir,
        num_augmentations_per_image=args.num_augmentations,
        target_size=args.target_size
    )


if __name__ == "__main__":
    main()
