"""
Global Keypoint Evaluator for all models.
Provides consistent evaluation metrics across different architectures.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from .metrics import calculate_mre, calculate_sdr


class KeypointEvaluator:
    """
    Global evaluator for keypoint detection models.
    
    Computes standard metrics:
    - MRE (Mean Radial Error)
    - SDR at multiple thresholds (2mm, 2.5mm, 3mm, 4mm)
    - Per-keypoint statistics
    """
    
    def __init__(
        self,
        sdr_thresholds: List[float] = [2.0, 2.5, 3.0, 4.0],
        pixel_spacing: float = 1.0  # mm per pixel
    ):
        """
        Initialize evaluator.
        
        Args:
            sdr_thresholds: List of thresholds for SDR calculation (in mm)
            pixel_spacing: Pixel spacing in mm (for physical distance calculation)
        """
        self.sdr_thresholds = sdr_thresholds
        self.pixel_spacing = pixel_spacing
    
    def extract_keypoints_from_heatmaps(
        self,
        heatmaps: torch.Tensor,
        threshold: float = 0.1
    ) -> List[np.ndarray]:
        """
        Extract keypoint coordinates from heatmaps.
        
        Args:
            heatmaps: Predicted heatmaps [B, K, H, W]
            threshold: Confidence threshold
            
        Returns:
            List of keypoint arrays for each image [N, 2] (x, y)
        """
        from scipy.ndimage import maximum_filter
        
        batch_size, num_keypoint_types, h, w = heatmaps.shape
        heatmaps_np = torch.sigmoid(heatmaps).cpu().numpy()
        
        batch_keypoints = []
        
        for b in range(batch_size):
            image_keypoints = []
            
            for k in range(num_keypoint_types):
                heatmap = heatmaps_np[b, k]
                
                # Find peaks using non-maximum suppression
                local_max = maximum_filter(heatmap, size=3) == heatmap
                peaks = (heatmap > threshold) & local_max
                
                # Get coordinates
                y_coords, x_coords = np.where(peaks)
                confidences = heatmap[peaks]
                
                if len(x_coords) > 0:
                    # Take the highest confidence peak
                    max_idx = np.argmax(confidences)
                    image_keypoints.append([x_coords[max_idx], y_coords[max_idx]])
                else:
                    # No detection - use center as fallback
                    image_keypoints.append([w // 2, h // 2])
            
            batch_keypoints.append(np.array(image_keypoints))
        
        return batch_keypoints
    
    def compute_metrics(
        self,
        pred_keypoints: List[np.ndarray],
        target_keypoints: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_keypoints: List of predicted keypoint arrays [N, 2]
            target_keypoints: List of target keypoint tensors [M, 4, 3] (vertebrae, corners, xyz)
            
        Returns:
            Dictionary of metrics
        """
        all_pred_points = []
        all_target_points = []
        
        # Flatten keypoints for comparison
        for pred, target in zip(pred_keypoints, target_keypoints):
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()
            
            # Target shape: [num_vertebrae, 4, 3] -> flatten to [num_vertebrae*4, 2]
            target_flat = target[:, :, :2].reshape(-1, 2)  # Take x, y only
            
            # Match predicted keypoints to targets
            # For UNet: pred is [4, 2] (4 keypoint types)
            # We need to match each pred keypoint type to all vertebrae of that type
            num_vertebrae = target.shape[0]
            
            for v in range(num_vertebrae):
                for k in range(4):  # 4 corners
                    if k < len(pred):
                        all_pred_points.append(pred[k])
                        all_target_points.append(target[v, k, :2])
        
        if len(all_pred_points) == 0:
            return {
                'MRE': 0.0,
                **{f'SDR_{int(t)}mm': 0.0 for t in self.sdr_thresholds}
            }
        
        pred_array = np.array(all_pred_points)
        target_array = np.array(all_target_points)
        
        # Calculate MRE
        mre = calculate_mre(pred_array, target_array)
        
        # Calculate SDR at multiple thresholds
        metrics = {'MRE': mre}
        for threshold in self.sdr_thresholds:
            # Convert mm to pixels
            threshold_pixels = threshold / self.pixel_spacing
            sdr = calculate_sdr(pred_array, target_array, threshold_pixels)
            metrics[f'SDR_{threshold}mm'] = sdr
        
        return metrics
    
    def evaluate_batch(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_keypoints: List[torch.Tensor],
        extract_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
            target_keypoints: Ground truth keypoints
            extract_threshold: Threshold for keypoint extraction
            
        Returns:
            Dictionary of metrics for this batch
        """
        # Extract keypoints from predicted heatmaps
        pred_keypoints = self.extract_keypoints_from_heatmaps(
            pred_heatmaps,
            threshold=extract_threshold
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
        sdr_thresholds: List of SDR thresholds in mm
        pixel_spacing: Pixel spacing in mm
        
    Returns:
        Global KeypointEvaluator instance
    """
    global _global_evaluator
    
    if _global_evaluator is None:
        if sdr_thresholds is None:
            sdr_thresholds = [2.0, 2.5, 3.0, 4.0]
        _global_evaluator = KeypointEvaluator(
            sdr_thresholds=sdr_thresholds,
            pixel_spacing=pixel_spacing
        )
    
    return _global_evaluator
