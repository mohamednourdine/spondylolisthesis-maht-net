"""
Abstract base trainer class for all models.
Provides common training functionality and structure.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
from datetime import datetime


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.
    
    Implements common training functionality:
    - Training/validation loops
    - Model checkpointing
    - Metrics tracking
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        save_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize base trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Configuration dictionary
            save_dir: Directory to save models and logs
            scheduler: Learning rate scheduler (optional)
            experiment_name: Name for this experiment (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.experiment_name = experiment_name
        
        # Setup save directory with better organization
        self.save_dir = Path(save_dir)
        if experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Format: model_name/experiment_name_timestamp
            model_name = config.get('model_name', 'unknown_model')
            self.save_dir = self.save_dir / model_name / f"{experiment_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mre = float('inf')  # Track best MRE for model selection
        self.best_val_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Early stopping
        self.patience = config.get('training', {}).get('early_stopping_patience', 10)
        self.patience_counter = 0
        
        # Save config
        self._save_config()
    
    def _save_config(self):
        """Save configuration to JSON."""
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Metrics to save
            is_best: Whether this is the best model
            filename: Custom filename (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_mre': self.best_val_mre,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_mre = checkpoint.get('best_val_mre', float('inf'))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def should_stop_early(self, val_metric: float, mode: str = 'min') -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_metric: Validation metric to check
            mode: 'min' for loss, 'max' for accuracy/metrics
            
        Returns:
            True if training should stop
        """
        improved = False
        
        if mode == 'min':
            if val_metric < self.best_val_loss:
                self.best_val_loss = val_metric
                improved = True
        else:  # mode == 'max'
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                improved = True
        
        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def train(self, num_epochs: int, resume_from: Optional[Path] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.save_dir.name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {start_epoch} -> {num_epochs}")
        print(f"Save directory: {self.save_dir}\n")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Print additional metrics in organized format
            train_other = {k: v for k, v in train_metrics.items() if k != 'loss'}
            val_other = {k: v for k, v in val_metrics.items() if k != 'loss'}
            
            if train_other:
                # Print MRE (prefer mm if available, otherwise px)
                if 'MRE_mm' in train_other:
                    print(f"  Train MRE: {train_other['MRE_mm']:.2f} mm ({train_other.get('MRE_px', 0):.1f} px)")
                elif 'MRE_px' in train_other:
                    print(f"  Train MRE: {train_other['MRE_px']:.2f} px")
                elif 'MRE' in train_other:  # Backward compatibility
                    print(f"  Train MRE: {train_other['MRE']:.2f}")
                # Print MSE if available
                if 'MSE_px' in train_other:
                    print(f"  Train MSE: {train_other['MSE_px']:.2f} px²")
                # Print SDR metrics in ascending order (2px, 4px, 8px, 16px)
                sdr_metrics = {k: v for k, v in train_other.items() if 'SDR' in k}
                if sdr_metrics:
                    # Sort by extracting numeric value from key (e.g., "SDR_2px" -> 2)
                    def get_threshold_value(key):
                        # Extract number from keys like "SDR_2px", "SDR_4px", etc.
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', key)
                        return float(match.group(1)) if match else float('inf')
                    
                    sorted_sdr = sorted(sdr_metrics.items(), key=lambda x: get_threshold_value(x[0]))
                    sdr_str = ', '.join([f"{k.replace('SDR_', '')}: {v:.4f}" for k, v in sorted_sdr])
                    print(f"  Train SDR: {sdr_str}")
                # Print any other metrics (excluding already printed ones)
                other = {k: v for k, v in train_other.items() if k not in ['MRE', 'MRE_px', 'MRE_mm', 'MSE_px'] and 'SDR' not in k}
                for key, value in other.items():
                    print(f"  Train {key}: {value:.4f}")
            
            if val_other:
                # Print MRE (prefer mm if available, otherwise px)
                if 'MRE_mm' in val_other:
                    print(f"  Val MRE:   {val_other['MRE_mm']:.2f} mm ({val_other.get('MRE_px', 0):.1f} px)")
                elif 'MRE_px' in val_other:
                    print(f"  Val MRE:   {val_other['MRE_px']:.2f} px")
                elif 'MRE' in val_other:  # Backward compatibility
                    print(f"  Val MRE:   {val_other['MRE']:.2f}")
                # Print MSE if available
                if 'MSE_px' in val_other:
                    print(f"  Val MSE:   {val_other['MSE_px']:.2f} px²")
                # Print SDR metrics in ascending order (2px, 4px, 8px, 16px)
                sdr_metrics = {k: v for k, v in val_other.items() if 'SDR' in k}
                if sdr_metrics:
                    # Sort by extracting numeric value from key (e.g., "SDR_2px" -> 2)
                    import re
                    def get_threshold_value(key):
                        match = re.search(r'(\d+(?:\.\d+)?)', key)
                        return float(match.group(1)) if match else float('inf')
                    
                    sorted_sdr = sorted(sdr_metrics.items(), key=lambda x: get_threshold_value(x[0]))
                    sdr_str = ', '.join([f"{k.replace('SDR_', '')}: {v:.4f}" for k, v in sorted_sdr])
                    print(f"  Val SDR:   {sdr_str}")
                # Print any other metrics (excluding already printed ones)
                other = {k: v for k, v in val_other.items() if k not in ['MRE', 'MRE_px', 'MRE_mm', 'MSE_px'] and 'SDR' not in k}
                for key, value in other.items():
                    print(f"  Val {key}: {value:.4f}")
            
            # Save best model based on MRE (the metric we care about)
            # Fall back to loss if MRE not available
            current_mre = val_metrics.get('MRE_px', val_metrics.get('MRE', float('inf')))
            is_best = current_mre < self.best_val_mre
            
            if is_best:
                self.best_val_mre = current_mre
                self.best_val_loss = val_metrics['loss']  # Also track loss
                self.patience_counter = 0  # Reset patience counter
                print(f"  ✓ New best model (val_MRE: {current_mre:.2f} px, val_loss: {val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, filename=f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {self.patience} epochs")
                break
        
        # Training complete
        total_time = time.time() - training_start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.save_dir}")
        
        # Save final training history
        self._save_training_history()
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
        }
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
