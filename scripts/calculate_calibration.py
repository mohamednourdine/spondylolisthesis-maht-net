#!/usr/bin/env python
"""
Calculate pixel spacing from real dataset images.
Analyzes vertebra dimensions to derive calibration.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np

def calculate_pixel_spacing():
    """Calculate pixel spacing from dataset."""
    
    # Paths
    train_img_dir = Path('data/Train/Keypointrcnn_data/images/train')
    train_label_dir = Path('data/Train/Keypointrcnn_data/labels/train')
    
    # Get sample files (analyze 20 images for better statistics)
    img_files = sorted(list(train_img_dir.glob('*.jpg')))[:20]
    label_files = sorted(list(train_label_dir.glob('*.json')))[:20]
    
    print('Analyzing Real Dataset Images:')
    print('='*60)
    print()
    
    # Collect measurements
    image_sizes = []
    vertebra_heights = []
    vertebra_widths = []
    
    for img_file, label_file in zip(img_files, label_files):
        # Load image
        img = Image.open(img_file)
        width, height = img.size
        image_sizes.append((width, height))
        
        # Load annotation
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        # Calculate vertebra dimensions
        keypoints = np.array(data['keypoints'])  # [N_vertebrae, 4_corners, 3]
        
        for vert_kp in keypoints:
            if len(vert_kp) == 4:  # Has 4 corners
                # Height: distance between top and bottom corners
                top_y = (vert_kp[0, 1] + vert_kp[1, 1]) / 2
                bottom_y = (vert_kp[2, 1] + vert_kp[3, 1]) / 2
                vert_height_px = abs(bottom_y - top_y)
                
                # Width: distance between left and right corners
                left_x = (vert_kp[0, 0] + vert_kp[2, 0]) / 2
                right_x = (vert_kp[1, 0] + vert_kp[3, 0]) / 2
                vert_width_px = abs(right_x - left_x)
                
                if vert_height_px > 0:
                    vertebra_heights.append(vert_height_px)
                if vert_width_px > 0:
                    vertebra_widths.append(vert_width_px)
    
    print(f'Images analyzed: {len(img_files)}')
    print(f'Vertebrae measured: {len(vertebra_heights)}')
    print(f'Image dimensions: {image_sizes[0]}')
    print()
    
    # Statistics
    avg_height = np.mean(vertebra_heights)
    std_height = np.std(vertebra_heights)
    avg_width = np.mean(vertebra_widths)
    std_width = np.std(vertebra_widths)
    
    print('Vertebra Measurements (in pixels):')
    print(f'  Height: {avg_height:.1f} ± {std_height:.1f} px (range: {min(vertebra_heights):.1f} - {max(vertebra_heights):.1f})')
    print(f'  Width:  {avg_width:.1f} ± {std_width:.1f} px (range: {min(vertebra_widths):.1f} - {max(vertebra_widths):.1f})')
    print()
    
    # Clinical reference values for lumbar vertebrae
    # From literature: L1-L5 vertebral body height ranges 25-30mm, average ~27mm
    # Width ranges 35-50mm, average ~42mm
    typical_height_mm = 27.0  # Conservative estimate
    typical_width_mm = 42.0
    
    # Calculate spacing using both measurements
    spacing_from_height = typical_height_mm / avg_height
    spacing_from_width = typical_width_mm / avg_width
    
    # Use average of both
    calculated_spacing = (spacing_from_height + spacing_from_width) / 2
    
    print('Calibration Calculation:')
    print(f'  Clinical reference: {typical_height_mm}mm height, {typical_width_mm}mm width')
    print(f'  From height: {spacing_from_height:.3f} mm/px ({1/spacing_from_height:.2f} px/mm)')
    print(f'  From width:  {spacing_from_width:.3f} mm/px ({1/spacing_from_width:.2f} px/mm)')
    print(f'  Average:     {calculated_spacing:.3f} mm/px ({1/calculated_spacing:.2f} px/mm)')
    print()
    
    print('Comparison with Current Config:')
    current_spacing = 0.167
    print(f'  Current: {current_spacing:.3f} mm/px ({1/current_spacing:.1f} px/mm)')
    print(f'  Calculated: {calculated_spacing:.3f} mm/px ({1/calculated_spacing:.1f} px/mm)')
    difference_pct = abs(calculated_spacing - current_spacing) / current_spacing * 100
    print(f'  Difference: {difference_pct:.1f}%')
    print()
    
    # Recommend spacing
    recommended_spacing = round(calculated_spacing, 3)
    print(f'Recommended pixel_spacing_mm: {recommended_spacing}')
    print(f'This means: 1mm = {1/recommended_spacing:.1f} pixels')
    
    return recommended_spacing, image_sizes[0]


if __name__ == '__main__':
    spacing, img_size = calculate_pixel_spacing()
