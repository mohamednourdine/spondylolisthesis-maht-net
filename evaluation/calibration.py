"""
Pixel spacing calibration utilities for medical imaging.

Handles conversion between pixel coordinates and physical measurements (mm).
Supports multiple calibration methods following clinical standards.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PixelSpacingCalibrator:
    """
    Manages pixel spacing calibration for medical images.
    
    Supports multiple calibration sources:
    1. EXIF/metadata from images
    2. JSON configuration files
    3. Manual specification
    4. Anatomical reference measurements
    """
    
    def __init__(
        self,
        default_spacing: Optional[float] = None,
        config_file: Optional[Path] = None
    ):
        """
        Initialize calibrator.
        
        Args:
            default_spacing: Default pixel spacing in mm/pixel (if no other source available)
            config_file: Path to JSON file with calibration data
        """
        self.default_spacing = default_spacing
        self.image_spacings = {}  # Per-image calibration
        
        if config_file and config_file.exists():
            self.load_calibration_config(config_file)
    
    def load_calibration_config(self, config_file: Path):
        """
        Load calibration from JSON configuration.
        
        Expected format:
        {
            "default_pixel_spacing_mm": 0.167,
            "images": {
                "image1.jpg": {"spacing": 0.165, "source": "manual"},
                "image2.jpg": {"spacing": 0.170, "source": "dicom"}
            }
        }
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'default_pixel_spacing_mm' in config:
                self.default_spacing = config['default_pixel_spacing_mm']
            
            if 'images' in config:
                for img_name, img_data in config['images'].items():
                    self.image_spacings[img_name] = img_data['spacing']
            
            logger.info(f"Loaded calibration config from {config_file}")
            logger.info(f"  Default spacing: {self.default_spacing} mm/pixel")
            logger.info(f"  Image-specific spacings: {len(self.image_spacings)}")
        
        except Exception as e:
            logger.error(f"Error loading calibration config: {e}")
    
    def get_pixel_spacing(
        self,
        image_path: Optional[Union[str, Path]] = None
    ) -> Optional[float]:
        """
        Get pixel spacing for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Pixel spacing in mm/pixel, or None if unavailable
        """
        # Check per-image calibration
        if image_path:
            image_name = Path(image_path).name
            if image_name in self.image_spacings:
                return self.image_spacings[image_name]
        
        # Use default
        return self.default_spacing
    
    def set_pixel_spacing(
        self,
        spacing: float,
        image_path: Optional[Union[str, Path]] = None
    ):
        """
        Set pixel spacing for image(s).
        
        Args:
            spacing: Pixel spacing in mm/pixel
            image_path: If provided, set for specific image; otherwise set default
        """
        if image_path:
            image_name = Path(image_path).name
            self.image_spacings[image_name] = spacing
        else:
            self.default_spacing = spacing
    
    def pixels_to_mm(
        self,
        pixel_distance: float,
        image_path: Optional[Union[str, Path]] = None
    ) -> Optional[float]:
        """
        Convert pixel distance to millimeters.
        
        Args:
            pixel_distance: Distance in pixels
            image_path: Optional image path for per-image calibration
            
        Returns:
            Distance in mm, or None if calibration unavailable
        """
        spacing = self.get_pixel_spacing(image_path)
        if spacing is None:
            return None
        return pixel_distance * spacing
    
    def mm_to_pixels(
        self,
        mm_distance: float,
        image_path: Optional[Union[str, Path]] = None
    ) -> Optional[float]:
        """
        Convert millimeter distance to pixels.
        
        Args:
            mm_distance: Distance in mm
            image_path: Optional image path for per-image calibration
            
        Returns:
            Distance in pixels, or None if calibration unavailable
        """
        spacing = self.get_pixel_spacing(image_path)
        if spacing is None:
            return None
        return mm_distance / spacing
    
    def estimate_spacing_from_annotations(
        self,
        annotations: Dict,
        known_dimension_mm: float,
        dimension_type: str = 'vertebra_height'
    ) -> float:
        """
        Estimate pixel spacing from known anatomical dimensions.
        
        Useful for retrospective calibration when DICOM data unavailable.
        
        Args:
            annotations: Annotation data with keypoints
            known_dimension_mm: Known physical dimension in mm
            dimension_type: Type of dimension ('vertebra_height', 'vertebra_width', etc.)
            
        Returns:
            Estimated pixel spacing in mm/pixel
        """
        # Example: Average lumbar vertebra height is ~28-32mm
        # This would need to be implemented based on specific annotation format
        raise NotImplementedError("Anatomical calibration not yet implemented")
    
    def save_calibration_config(self, config_file: Path):
        """Save calibration configuration to JSON file."""
        config = {
            'default_pixel_spacing_mm': self.default_spacing,
            'images': {
                name: {'spacing': spacing, 'source': 'manual'}
                for name, spacing in self.image_spacings.items()
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved calibration config to {config_file}")


def estimate_dataset_pixel_spacing(
    image_dir: Path,
    annotation_dir: Path,
    method: str = 'automatic'
) -> Tuple[float, float]:
    """
    Estimate pixel spacing statistics for a dataset.
    
    Args:
        image_dir: Directory containing images
        annotation_dir: Directory containing annotations
        method: Estimation method ('automatic', 'manual', 'reference')
        
    Returns:
        (mean_spacing, std_spacing) in mm/pixel
    """
    # This would analyze the dataset and estimate spacing
    # For now, return placeholder values
    logger.warning("Automatic spacing estimation not implemented")
    logger.info("Consider manual calibration or DICOM metadata extraction")
    
    # Typical values for spine X-rays:
    # - Digital radiography: 0.1-0.2 mm/pixel
    # - CT: 0.5-1.0 mm/pixel  
    # - MRI: 0.5-2.0 mm/pixel
    
    return None, None


def create_default_calibration_config(
    output_file: Path,
    dataset_type: str = 'spine_xray'
) -> Path:
    """
    Create a template calibration configuration file.
    
    Args:
        output_file: Where to save the config
        dataset_type: Type of medical imaging dataset
        
    Returns:
        Path to created config file
    """
    # Typical values for different imaging modalities
    default_spacings = {
        'spine_xray': 0.167,  # ~6 pixels per mm (common for digital radiography)
        'ct': 0.75,
        'mri': 1.0,
        'ultrasound': 0.1
    }
    
    config = {
        '_comment': 'Pixel spacing calibration configuration',
        '_units': 'mm per pixel',
        'default_pixel_spacing_mm': default_spacings.get(dataset_type, 0.167),
        'dataset_info': {
            'type': dataset_type,
            'source': 'manual_configuration',
            'calibration_method': 'Please update with actual calibration data'
        },
        'images': {
            'example_image.jpg': {
                'spacing': default_spacings.get(dataset_type, 0.167),
                'source': 'template',
                'notes': 'Update with actual values'
            }
        },
        'instructions': {
            '1': 'Set default_pixel_spacing_mm based on your imaging equipment',
            '2': 'Add per-image calibration if spacing varies',
            '3': 'Document calibration method for reproducibility',
            '4': 'Common spine X-ray values: 0.14-0.20 mm/pixel'
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created calibration config template: {output_file}")
    logger.info("Please update with actual calibration values before use")
    
    return output_file
