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
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch in pbar:
            images = batch['images'].to(self.device)
            target_heatmaps = batch['heatmaps'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_heatmaps = self.model(images)
            
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
            
            # Update progress bar
            if batch_metrics_list:
                last_metrics = batch_metrics_list[-1]
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'MRE': f'{last_metrics["MRE"]:.1f}',
                    'SDR_2mm': f'{last_metrics["SDR_2.0mm"]:.3f}'
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Aggregate metrics
        metrics = {'loss': running_loss / num_batches}
        
        if batch_metrics_list:
            aggregated_metrics = self.evaluator.aggregate_metrics(batch_metrics_list)
            metrics.update(aggregated_metrics)
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        batch_metrics_list = []
        num_batches = 0
        
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
                
                # Update progress bar
                if batch_metrics_list:
                    last_metrics = batch_metrics_list[-1]
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'MRE': f'{last_metrics["MRE"]:.1f}',
                        'SDR_2mm': f'{last_metrics["SDR_2.0mm"]:.3f}'
                    })
                else:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Aggregate metrics
        metrics = {'loss': running_loss / num_batches}
        
        if batch_metrics_list:
            aggregated_metrics = self.evaluator.aggregate_metrics(batch_metrics_list)
            metrics.update(aggregated_metrics)
        
        return metrics
