#!/usr/bin/env python
"""
Interactive calibration tool.
Allows manual measurement of known distances to calculate pixel spacing.

Usage:
1. Find an X-ray with a calibration marker or known anatomical distance
2. Measure the distance in pixels
3. Enter the actual distance in mm
4. Calculate pixel spacing
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def measure_distance_interactive():
    """Interactive tool to measure pixel spacing from known distances."""
    
    print('='*70)
    print('INTERACTIVE PIXEL SPACING CALIBRATION')
    print('='*70)
    print()
    print('This tool helps calculate pixel spacing from known measurements.')
    print('You need:')
    print('  1. An X-ray image from your dataset')
    print('  2. A known real-world distance visible in the image')
    print('     (e.g., calibration marker, known anatomical distance)')
    print()
    
    # Option 1: Does the dataset have calibration markers?
    print('Question 1: Do your X-ray images have calibration markers?')
    print('  (Rulers, balls of known size, or reference objects)')
    has_marker = input('Enter yes/no: ').strip().lower()
    print()
    
    if has_marker.startswith('y'):
        print('Great! You can measure the marker to get accurate pixel spacing.')
        print()
        print('Steps:')
        print('  1. Open an image in an image viewer (Preview, GIMP, etc.)')
        print('  2. Measure the calibration marker in pixels')
        print('  3. Enter the measurements below')
        print()
        
        known_mm = float(input('Enter the known distance in mm: '))
        measured_px = float(input('Enter the measured distance in pixels: '))
        
        calculated_spacing = known_mm / measured_px
        print()
        print(f'Calculated pixel spacing: {calculated_spacing:.4f} mm/px')
        print(f'This means: 1mm = {1/calculated_spacing:.2f} pixels')
        
        return calculated_spacing
    
    else:
        print('No calibration markers detected.')
        print()
        print('Alternative method: Statistical analysis from vertebra dimensions')
        print()
        print('The script already calculated spacing from vertebra dimensions:')
        print('  - Measured vertebra heights in pixels from annotations')
        print('  - Assumed typical lumbar vertebra dimensions from literature')
        print('  - Result: 0.609 mm/px')
        print()
        print('IMPORTANT: This assumes average vertebra sizes!')
        print('For accurate calibration, you should:')
        print()
        print('Option A: Find DICOM metadata')
        print('  - Original DICOM files contain PixelSpacing tag')
        print('  - Check if you have access to original medical imaging files')
        print()
        print('Option B: Contact imaging facility')
        print('  - Ask for pixel spacing or calibration information')
        print('  - Typical spine X-rays: 0.14-0.20 mm/px for CR/DR systems')
        print()
        print('Option C: Use current estimate (0.609 mm/px)')
        print('  - Based on measured vertebra dimensions')
        print('  - Reasonable for relative measurements')
        print('  - May not be accurate for absolute distances')
        print()
        
        choice = input('Use current estimate? (yes/no): ').strip().lower()
        if choice.startswith('y'):
            return 0.609
        else:
            manual_spacing = float(input('Enter pixel spacing manually (mm/px): '))
            return manual_spacing


def check_dicom_metadata():
    """Check if DICOM files are available."""
    print()
    print('='*70)
    print('CHECKING FOR DICOM METADATA')
    print('='*70)
    print()
    
    # Look for DICOM files
    data_dir = Path('data')
    dicom_files = list(data_dir.rglob('*.dcm')) + list(data_dir.rglob('*.dicom'))
    
    if dicom_files:
        print(f'Found {len(dicom_files)} DICOM files!')
        print('DICOM files contain accurate PixelSpacing metadata.')
        print()
        print('To extract pixel spacing:')
        print('  pip install pydicom')
        print('  python -c "import pydicom; ds=pydicom.dcmread(\'file.dcm\'); print(ds.PixelSpacing)"')
        return True
    else:
        print('No DICOM files found in data directory.')
        print('Current images appear to be converted (JPG format).')
        print('Pixel spacing metadata may have been lost during conversion.')
        return False


def analyze_current_calibration():
    """Show current calibration details."""
    print()
    print('='*70)
    print('CURRENT CALIBRATION ANALYSIS')
    print('='*70)
    print()
    
    calib_file = Path('config/pixel_spacing_calibration.json')
    with open(calib_file) as f:
        calib = json.load(f)
    
    spacing = calib['default_pixel_spacing_mm']
    details = calib['dataset_info']['analysis_details']
    
    print(f'Current pixel spacing: {spacing} mm/px ({1/spacing:.2f} px/mm)')
    print()
    print('Calculation method:')
    print(f'  - Analyzed {details["images_analyzed"]} images')
    print(f'  - Measured {details["vertebrae_measured"]} vertebrae')
    print(f'  - Average vertebra height: {details["vertebra_height_px"]}')
    print(f'  - Average vertebra width: {details["vertebra_width_px"]}')
    print()
    print(f'  - Assumed typical vertebra height: {details["clinical_reference_height_mm"]} mm')
    print(f'  - Assumed typical vertebra width: {details["clinical_reference_width_mm"]} mm')
    print()
    print('Calculation: pixel_spacing = assumed_real_size_mm / measured_pixels')
    print(f'  From height: {details["calculated_from_height"]} mm/px')
    print(f'  From width:  {details["calculated_from_width"]} mm/px')
    print(f'  Average:     {spacing} mm/px')
    print()
    print('⚠️  WARNING: This assumes standard vertebra dimensions!')
    print('    - Patient population may vary (age, sex, ethnicity)')
    print('    - Pathology may affect vertebra sizes')
    print('    - X-ray magnification not accounted for')
    print()


if __name__ == '__main__':
    print()
    analyze_current_calibration()
    has_dicom = check_dicom_metadata()
    print()
    
    if not has_dicom:
        print()
        spacing = measure_distance_interactive()
        
        if spacing:
            print()
            print('='*70)
            print(f'FINAL CALIBRATION: {spacing:.4f} mm/px')
            print('='*70)
            print()
            print('To update the configuration:')
            print(f'  1. Edit config/pixel_spacing_calibration.json')
            print(f'  2. Set "default_pixel_spacing_mm": {spacing:.4f}')
            print(f'  3. Edit config/mac_config.py')
            print(f'  4. Set PIXEL_SPACING_MM = {spacing:.4f}')
