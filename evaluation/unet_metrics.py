"""
Evaluation metrics for UNet keypoint detection model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist


def calculate_pck(
    pred_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    threshold: float = 0.05,
    normalize_by: str = 'bbox'
) -> float:
    """
    Calculate Percentage of Correct Keypoints (PCK).
    
    Args:
        pred_keypoints: Predicted keypoints [N, 2] (x, y)
        target_keypoints: Target keypoints [N, 2] (x, y)
        threshold: Distance threshold as fraction of normalization factor
        normalize_by: 'bbox' or 'image' - how to normalize distances
        
    Returns:
        PCK score (0-1)
    """
    if len(pred_keypoints) == 0 or len(target_keypoints) == 0:
        return 0.0
    
    # Calculate distances
    if pred_keypoints.ndim == 1:
        pred_keypoints = pred_keypoints.reshape(1, -1)
    if target_keypoints.ndim == 1:
        target_keypoints = target_keypoints.reshape(1, -1)
    
    distances = np.sqrt(np.sum((pred_keypoints - target_keypoints) ** 2, axis=1))
    
    # Normalize
    if normalize_by == 'bbox':
        # Use max bbox dimension as normalization
        bbox_size = max(
            np.max(target_keypoints[:, 0]) - np.min(target_keypoints[:, 0]),
            np.max(target_keypoints[:, 1]) - np.min(target_keypoints[:, 1])
        )
        normalized_distances = distances / bbox_size
    else:  # 'image'
        # Assume image size (will be passed in real implementation)
        normalized_distances = distances / 512.0  # Default image size
    
    # Calculate PCK
    correct = (normalized_distances < threshold).sum()
    pck = correct / len(distances)
    
    return pck


def calculate_keypoint_distance_metrics(
    pred_keypoints: List[np.ndarray],
    target_keypoints: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calculate various distance-based metrics for keypoint prediction.
    
    Args:
        pred_keypoints: List of predicted keypoint arrays [N, 2]
        target_keypoints: List of target keypoint arrays [N, 2]
        
    Returns:
        Dictionary of metrics
    """
    all_distances = []
    all_pck_05 = []
    all_pck_10 = []
    all_pck_15 = []
    
    for pred, target in zip(pred_keypoints, target_keypoints):
        if len(pred) == 0 or len(target) == 0:
            continue
        
        # Calculate distances
        distances = np.sqrt(np.sum((pred - target) ** 2, axis=1))
        all_distances.extend(distances)
        
        # Calculate PCK at different thresholds
        pck_05 = calculate_pck(pred, target, threshold=0.05)
        pck_10 = calculate_pck(pred, target, threshold=0.10)
        pck_15 = calculate_pck(pred, target, threshold=0.15)
        
        all_pck_05.append(pck_05)
        all_pck_10.append(pck_10)
        all_pck_15.append(pck_15)
    
    metrics = {
        'mean_distance': np.mean(all_distances) if all_distances else 0.0,
        'median_distance': np.median(all_distances) if all_distances else 0.0,
        'std_distance': np.std(all_distances) if all_distances else 0.0,
        'pck_0.05': np.mean(all_pck_05) if all_pck_05 else 0.0,
        'pck_0.10': np.mean(all_pck_10) if all_pck_10 else 0.0,
        'pck_0.15': np.mean(all_pck_15) if all_pck_15 else 0.0,
    }
    
    return metrics


def extract_keypoints_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    nms_kernel_size: int = 3
) -> np.ndarray:
    """
    Extract keypoint locations from a heatmap.
    
    Args:
        heatmap: Heatmap array [H, W]
        threshold: Confidence threshold
        nms_kernel_size: Kernel size for non-maximum suppression
        
    Returns:
        Keypoints array [N, 3] (x, y, confidence)
    """
    from scipy.ndimage import maximum_filter
    
    # Apply non-maximum suppression
    local_max = maximum_filter(heatmap, size=nms_kernel_size) == heatmap
    peaks = (heatmap > threshold) & local_max
    
    # Get coordinates
    y_coords, x_coords = np.where(peaks)
    confidences = heatmap[peaks]
    
    if len(x_coords) == 0:
        return np.array([]).reshape(0, 3)
    
    # Stack as [x, y, confidence]
    keypoints = np.stack([x_coords, y_coords, confidences], axis=1)
    
    # Sort by confidence
    keypoints = keypoints[np.argsort(-keypoints[:, 2])]
    
    return keypoints


def match_keypoints(
    pred_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    max_distance: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predicted keypoints to target keypoints using Hungarian algorithm.
    
    Args:
        pred_keypoints: Predicted keypoints [N, 2 or 3]
        target_keypoints: Target keypoints [M, 2 or 3]
        max_distance: Maximum distance for matching
        
    Returns:
        matched_pred: Matched predicted keypoints
        matched_target: Matched target keypoints
        unmatched_pred_indices: Indices of unmatched predictions
    """
    from scipy.optimize import linear_sum_assignment
    
    if len(pred_keypoints) == 0 or len(target_keypoints) == 0:
        return (
            np.array([]).reshape(0, 2),
            np.array([]).reshape(0, 2),
            np.arange(len(pred_keypoints))
        )
    
    # Extract xy coordinates
    pred_xy = pred_keypoints[:, :2]
    target_xy = target_keypoints[:, :2]
    
    # Calculate cost matrix (Euclidean distances)
    cost_matrix = cdist(pred_xy, target_xy, metric='euclidean')
    
    # Apply Hungarian algorithm
    pred_indices, target_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out matches that are too far
    valid_matches = cost_matrix[pred_indices, target_indices] < max_distance
    pred_indices = pred_indices[valid_matches]
    target_indices = target_indices[valid_matches]
    
    # Get matched keypoints
    matched_pred = pred_xy[pred_indices]
    matched_target = target_xy[target_indices]
    
    # Get unmatched predictions
    all_pred_indices = set(range(len(pred_keypoints)))
    matched_pred_indices = set(pred_indices)
    unmatched_pred_indices = np.array(list(all_pred_indices - matched_pred_indices))
    
    return matched_pred, matched_target, unmatched_pred_indices


def evaluate_unet_predictions(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_keypoints: List[torch.Tensor],
    threshold: float = 0.5,
    image_size: Tuple[int, int] = (512, 512)
) -> Dict[str, float]:
    """
    Comprehensive evaluation of UNet predictions.
    
    Args:
        pred_heatmaps: Predicted heatmaps [B, K, H, W]
        target_heatmaps: Target heatmaps [B, K, H, W]
        target_keypoints: List of target keypoints for each image
        threshold: Threshold for keypoint extraction
        image_size: Original image size
        
    Returns:
        Dictionary of evaluation metrics
    """
    batch_size, num_keypoint_types, h, w = pred_heatmaps.shape
    
    # Convert to numpy
    pred_heatmaps_np = torch.sigmoid(pred_heatmaps).cpu().numpy()
    target_heatmaps_np = target_heatmaps.cpu().numpy()
    
    # Heatmap-level metrics
    mse = np.mean((pred_heatmaps_np - target_heatmaps_np) ** 2)
    mae = np.mean(np.abs(pred_heatmaps_np - target_heatmaps_np))
    
    # Extract keypoints and calculate distance metrics
    all_pred_keypoints = []
    all_target_keypoints = []
    
    for b in range(batch_size):
        # For each keypoint type, extract predicted locations
        for k in range(num_keypoint_types):
            pred_kps = extract_keypoints_from_heatmap(
                pred_heatmaps_np[b, k],
                threshold=threshold
            )
            
            if len(pred_kps) > 0:
                # Scale back to original image size
                scale_x = image_size[1] / w
                scale_y = image_size[0] / h
                pred_kps[:, 0] *= scale_x
                pred_kps[:, 1] *= scale_y
                all_pred_keypoints.append(pred_kps[:, :2])
            else:
                all_pred_keypoints.append(np.array([]).reshape(0, 2))
        
        # Get target keypoints for this image
        if b < len(target_keypoints):
            target_kps = target_keypoints[b].cpu().numpy()
            # Reshape to [N*4, 3] where N is number of vertebrae
            target_kps_flat = target_kps.reshape(-1, 3)
            all_target_keypoints.extend([kp[:2] for kp in target_kps_flat])
    
    # Calculate distance metrics
    if len(all_pred_keypoints) > 0 and len(all_target_keypoints) > 0:
        distance_metrics = calculate_keypoint_distance_metrics(
            all_pred_keypoints,
            all_target_keypoints
        )
    else:
        distance_metrics = {
            'mean_distance': 0.0,
            'median_distance': 0.0,
            'std_distance': 0.0,
            'pck_0.05': 0.0,
            'pck_0.10': 0.0,
            'pck_0.15': 0.0,
        }
    
    # Combine all metrics
    metrics = {
        'heatmap_mse': mse,
        'heatmap_mae': mae,
        **distance_metrics
    }
    
    return metrics


class UNetEvaluator:
    """Evaluator for UNet model."""
    
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
    
    def evaluate(self, dataloader, num_batches=None):
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: DataLoader for evaluation
            num_batches: Number of batches to evaluate (None = all)
            
        Returns:
            Dictionary of aggregated metrics
        """
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                
                images = batch['images'].to(self.device)
                target_heatmaps = batch['heatmaps'].to(self.device)
                target_keypoints = batch['keypoints']
                
                # Forward pass
                pred_heatmaps = self.model(images)
                
                # Evaluate
                metrics = evaluate_unet_predictions(
                    pred_heatmaps,
                    target_heatmaps,
                    target_keypoints,
                    threshold=self.threshold
                )
                
                all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = np.mean(values)
        
        return aggregated
