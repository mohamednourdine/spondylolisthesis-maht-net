"""
Global Keypoint Evaluator for all models.
Provides consistent evaluation metrics across different architectures.

Supports dual reporting (pixels and mm) for clinical applications.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from .metrics import calculate_mre, calculate_sdr
from .calibration import PixelSpacingCalibrator


class KeypointEvaluator:
    """
    Global evaluator for keypoint detection models.
    
    Computes standard metrics with dual reporting (pixels and mm):
    - MRE (Mean Radial Error)
    - SDR at multiple thresholds
    - Per-keypoint statistics
    
    Follows best practices for medical imaging publications:
    - Always reports pixel-based metrics (resolution-independent baseline)
    - Optionally reports physical metrics (mm) when calibration available
    - Supports per-image calibration for varied imaging protocols
    """
    
    def __init__(
        self,
        sdr_thresholds_px: List[float] = [2.0, 4.0, 8.0, 16.0],
        calibrator: Optional[PixelSpacingCalibrator] = None,
        report_physical_metrics: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            sdr_thresholds_px: SDR thresholds in pixels (always used)
            calibrator: PixelSpacingCalibrator for mm conversion (optional)
            report_physical_metrics: Whether to include mm metrics when calibration available
        """
        self.sdr_thresholds_px = sdr_thresholds_px
        self.calibrator = calibrator
        self.report_physical_metrics = report_physical_metrics
        
        # Compute mm thresholds if calibrator available
        self.sdr_thresholds_mm = None
        if calibrator and calibrator.default_spacing:
            self.sdr_thresholds_mm = [
                calibrator.pixels_to_mm(t) for t in sdr_thresholds_px
            ]
    
    def extract_keypoints_from_heatmaps(
        self,
        heatmaps: torch.Tensor,
        threshold: float = 0.1,
        max_vertebrae: int = 10,
        corners_per_vertebra: int = 4
    ) -> List[np.ndarray]:
        """
        Extract keypoint coordinates from per-vertebra heatmaps using argmax.
        
        Args:
            heatmaps: Predicted heatmaps [B, max_vertebrae*corners_per_vertebra, H, W]
            threshold: Not used - kept for compatibility
            max_vertebrae: Maximum number of vertebrae (default: 10)
            corners_per_vertebra: Corners per vertebra (default: 4)
            
        Returns:
            List of keypoint arrays for each image [N_vertebrae, corners_per_vertebra, 2]
            where N_vertebrae is detected based on non-zero heatmaps
        """
        batch_size, total_channels, h, w = heatmaps.shape
        heatmaps_np = heatmaps.cpu().numpy()
        
        batch_keypoints = []
        
        for b in range(batch_size):
            vertebrae_keypoints = []
            
            # Process each vertebra - extract all MAX_VERTEBRAE regardless of confidence
            # (visibility filtering happens during metric computation using ground truth)
            for vert_idx in range(max_vertebrae):
                vertebra_corners = []
                
                # Extract 4 corners for this vertebra
                for corner_idx in range(corners_per_vertebra):
                    channel_idx = vert_idx * corners_per_vertebra + corner_idx
                    heatmap = heatmaps_np[b, channel_idx]
                    
                    # Always extract using argmax (no threshold filtering)
                    # Model outputs raw values, so any threshold would be arbitrary
                    max_idx = np.argmax(heatmap)
                    y, x = np.unravel_index(max_idx, heatmap.shape)
                    
                    # Mark as visible - ground truth visibility used in metrics
                    vertebra_corners.append([float(x), float(y), 1.0])
                
                vertebrae_keypoints.append(vertebra_corners)
            
            # Convert to numpy array [max_vertebrae, 4, 3]
            batch_keypoints.append(np.array(vertebrae_keypoints))
        
        return batch_keypoints
    
    def compute_metrics(
        self,
        pred_keypoints: List[np.ndarray],
        target_keypoints: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for per-vertebra predictions.
        
        Args:
            pred_keypoints: List of predicted keypoint arrays [N_vertebrae, 4, 3] (vertebrae, corners, xyz)
            target_keypoints: List of target keypoint tensors [M_vertebrae, 4, 3] (vertebrae, corners, xyz)
            
        Returns:
            Dictionary of metrics
        """
        all_pred_points = []
        all_target_points = []
        
        # Flatten keypoints for comparison
        for pred, target in zip(pred_keypoints, target_keypoints):
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()
            
            # Both pred and target have shape: [num_vertebrae, 4, 3]
            # Direct correspondence: vertebra i in pred matches vertebra i in target
            num_target_vertebrae = target.shape[0]
            num_pred_vertebrae = pred.shape[0]
            
            # Match vertebrae by index (both are sorted top-to-bottom)
            for v in range(min(num_target_vertebrae, num_pred_vertebrae)):
                for k in range(4):  # 4 corners
                    # Only count if both visible
                    if pred[v, k, 2] > 0.5 and target[v, k, 2] > 0.5:
                        all_pred_points.append(pred[v, k, :2])
                        all_target_points.append(target[v, k, :2])
        
        if len(all_pred_points) == 0:
            return {
                'MRE': 0.0,
                **{f'SDR_{int(t)}mm': 0.0 for t in self.sdr_thresholds}
            }
        
        pred_array = np.array(all_pred_points)
        target_array = np.array(all_target_points)
        
        # Calculate MRE in pixels (always)
        mre_px = calculate_mre(pred_array, target_array)
        
        # Calculate SDR at pixel thresholds (always)
        metrics = {'MRE_px': mre_px}
        for threshold_px in self.sdr_thresholds_px:
            sdr = calculate_sdr(pred_array, target_array, threshold_px)
            threshold_str = f'{int(threshold_px)}px' if threshold_px == int(threshold_px) else f'{threshold_px}px'
            metrics[f'SDR_{threshold_str}'] = sdr
        
        # Add physical metrics (mm) if calibration available
        if self.report_physical_metrics and self.calibrator and self.calibrator.default_spacing:
            mre_mm = self.calibrator.pixels_to_mm(mre_px)
            if mre_mm is not None:
                metrics['MRE_mm'] = mre_mm
                
                # SDR at mm thresholds
                if self.sdr_thresholds_mm:
                    for threshold_mm, threshold_px in zip(self.sdr_thresholds_mm, self.sdr_thresholds_px):
                        sdr = calculate_sdr(pred_array, target_array, threshold_px)
                        threshold_str = f'{threshold_mm:.1f}mm' if threshold_mm else f'{int(threshold_px)}px'
                        metrics[f'SDR_{threshold_str}'] = sdr
        
        return metrics
    
    def evaluate_batch(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_keypoints: List[torch.Tensor],
        extract_threshold: float = 0.1,
        max_vertebrae: int = 10,
        corners_per_vertebra: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, max_vertebrae*corners_per_vertebra, H, W]
            target_heatmaps: Target heatmaps [B, max_vertebrae*corners_per_vertebra, H, W]
            target_keypoints: Ground truth keypoints
            extract_threshold: Threshold for keypoint extraction
            max_vertebrae: Maximum number of vertebrae (default: 10)
            corners_per_vertebra: Corners per vertebra (default: 4)
            
        Returns:
            Dictionary of metrics for this batch
        """
        # Extract keypoints from predicted heatmaps
        pred_keypoints = self.extract_keypoints_from_heatmaps(
            pred_heatmaps,
            threshold=extract_threshold,
            max_vertebrae=max_vertebrae,
            corners_per_vertebra=corners_per_vertebra
        )
        
        # Compute metrics
        metrics = self.compute_metrics(pred_keypoints, target_keypoints)
        
        return metrics
    
    def aggregate_metrics(
        self,
        batch_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple batches.
        
        Args:
            batch_metrics: List of metric dictionaries from each batch
            
        Returns:
            Averaged metrics
        """
        if not batch_metrics:
            return {}
        
        aggregated = {}
        keys = batch_metrics[0].keys()
        
        for key in keys:
            values = [m[key] for m in batch_metrics if key in m]
            aggregated[key] = sum(values) / len(values) if values else 0.0
        
        return aggregated
    
    def format_metrics(self, metrics: Dict[str, float], prefix: str = '') -> str:
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names (e.g., 'train_' or 'val_')
            
        Returns:
            Formatted string
        """
        parts = []
        for key, value in metrics.items():
            if 'SDR' in key:
                parts.append(f"{prefix}{key}: {value:.4f}")
            else:
                parts.append(f"{prefix}{key}: {value:.2f}")
        return ', '.join(parts)


# Global evaluator instance
_global_evaluator = None


def get_global_evaluator(
    sdr_thresholds: Optional[List[float]] = None,
    pixel_spacing: float = 1.0
) -> KeypointEvaluator:
    """
    Get or create global evaluator instance.
    
    Args:
        sdr_thresholds: List of SDR thresholds in pixels (default: [2, 4, 8, 16])
        pixel_spacing: Deprecated - use calibrator instead
        
    Returns:
        Global KeypointEvaluator instance
    """
    global _global_evaluator
    
    if _global_evaluator is None:
        if sdr_thresholds is None:
            sdr_thresholds = [2.0, 4.0, 8.0, 16.0]
        _global_evaluator = KeypointEvaluator(
            sdr_thresholds_px=sdr_thresholds,
            calibrator=None,
            report_physical_metrics=False
        )
    
    return _global_evaluator
