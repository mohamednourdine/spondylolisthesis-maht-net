"""
UNet-specific trainer implementation.
"""

import torch
from tqdm import tqdm
from typing import Dict
from .base_trainer import BaseTrainer
from evaluation.keypoint_evaluator import get_global_evaluator


class UNetTrainer(BaseTrainer):
    """Trainer for UNet keypoint detection model."""
    
    def __init__(self, *args, compute_metrics: bool = True, **kwargs):
        """
        Initialize UNet trainer.
        
        Args:
            compute_metrics: Whether to compute keypoint metrics during training
            *args, **kwargs: Arguments for BaseTrainer
        """
        super().__init__(*args, **kwargs)
        self.compute_metrics = compute_metrics
        if self.compute_metrics:
            self.evaluator = get_global_evaluator()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        batch_metrics_list = []
        num_batches = 0
        
        # Statistics tracking
        heatmap_stats = {'pred_min': [], 'pred_max': [], 'pred_mean': [],
                        'target_min': [], 'target_max': [], 'target_mean': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            target_heatmaps = batch['heatmaps'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_heatmaps = self.model(images)
            
            # Track heatmap statistics (every 50 batches)
            if batch_idx % 50 == 0:
                heatmap_stats['pred_min'].append(pred_heatmaps.min().item())
                heatmap_stats['pred_max'].append(pred_heatmaps.max().item())
                heatmap_stats['pred_mean'].append(pred_heatmaps.mean().item())
                heatmap_stats['target_min'].append(target_heatmaps.min().item())
                heatmap_stats['target_max'].append(target_heatmaps.max().item())
                heatmap_stats['target_mean'].append(target_heatmaps.mean().item())
            
            # Compute loss
            loss = self.criterion(pred_heatmaps, target_heatmaps)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Clear cache to prevent memory buildup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Compute keypoint metrics (less frequently to save time)
            if self.compute_metrics and num_batches % 10 == 0:
                with torch.no_grad():
                    batch_metrics = self.evaluator.evaluate_batch(
                        pred_heatmaps.detach(),
                        target_heatmaps,
                        batch['keypoints']
                    )
                    batch_metrics_list.append(batch_metrics)
            
            # Update progress bar with heatmap ranges
            if batch_metrics_list:
                last_metrics = batch_metrics_list[-1]
                mre_key = 'MRE_mm' if 'MRE_mm' in last_metrics else 'MRE_px'
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'MRE': f'{last_metrics[mre_key]:.1f}',
                    'SDR_2px': f'{last_metrics["SDR_2px"]:.3f}',
                    'pred_range': f'[{pred_heatmaps.min().item():.1f},{pred_heatmaps.max().item():.1f}]'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pred_range': f'[{pred_heatmaps.min().item():.1f},{pred_heatmaps.max().item():.1f}]'
                })
        
        # Aggregate metrics
        metrics = {'loss': running_loss / num_batches}
        
        if batch_metrics_list:
            aggregated_metrics = self.evaluator.aggregate_metrics(batch_metrics_list)
            metrics.update(aggregated_metrics)
        
        # Print detailed statistics
        if heatmap_stats['pred_mean']:
            print(f"\n  Heatmap Statistics (Train):")
            print(f"    Predictions: min={min(heatmap_stats['pred_min']):.2f}, max={max(heatmap_stats['pred_max']):.2f}, mean={sum(heatmap_stats['pred_mean'])/len(heatmap_stats['pred_mean']):.2f}")
            print(f"    Targets:     min={min(heatmap_stats['target_min']):.2f}, max={max(heatmap_stats['target_max']):.2f}, mean={sum(heatmap_stats['target_mean'])/len(heatmap_stats['target_mean']):.2f}")
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        batch_metrics_list = []
        num_batches = 0
        per_sample_errors = []  # Track individual sample errors
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch in pbar:
                images = batch['images'].to(self.device)
                target_heatmaps = batch['heatmaps'].to(self.device)
                
                # Forward pass
                pred_heatmaps = self.model(images)
                
                # Compute loss
                loss = self.criterion(pred_heatmaps, target_heatmaps)
                
                running_loss += loss.item()
                num_batches += 1
                
                # Compute keypoint metrics
                if self.compute_metrics:
                    batch_metrics = self.evaluator.evaluate_batch(
                        pred_heatmaps,
                        target_heatmaps,
                        batch['keypoints']
                    )
                    batch_metrics_list.append(batch_metrics)
                    
                    # Track per-sample MRE for detailed analysis
                    mre_key = 'MRE_mm' if 'MRE_mm' in batch_metrics else 'MRE_px'
                    per_sample_errors.append(batch_metrics[mre_key])
                
                # Update progress bar
                if batch_metrics_list:
                    last_metrics = batch_metrics_list[-1]
                    mre_key = 'MRE_mm' if 'MRE_mm' in last_metrics else 'MRE_px'
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'MRE': f'{last_metrics[mre_key]:.1f}',
                        'SDR_2px': f'{last_metrics["SDR_2px"]:.3f}',
                        'pred_range': f'[{pred_heatmaps.min().item():.1f},{pred_heatmaps.max().item():.1f}]'
                    })
                else:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Aggregate metrics
        metrics = {'loss': running_loss / num_batches}
        
        if batch_metrics_list:
            aggregated_metrics = self.evaluator.aggregate_metrics(batch_metrics_list)
            metrics.update(aggregated_metrics)
            
            # Print per-sample error distribution
            if per_sample_errors:
                per_sample_errors.sort()
                print(f"\n  Per-Sample Error Distribution (Val):")
                print(f"    Best (min):      {per_sample_errors[0]:.2f} px")
                print(f"    Median:          {per_sample_errors[len(per_sample_errors)//2]:.2f} px")
                print(f"    Worst (max):     {per_sample_errors[-1]:.2f} px")
                print(f"    90th percentile: {per_sample_errors[int(len(per_sample_errors)*0.9)]:.2f} px")
        
        return metrics
