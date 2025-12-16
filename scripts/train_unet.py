"""
Training script for UNet keypoint detection model.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.unet import create_unet
from src.data.unet_dataset import create_unet_dataloaders
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import SpondylolisthesisAugmentation


class UNetKeypointLoss(nn.Module):
    """
    Loss function for UNet keypoint detection.
    Combines MSE loss with optional focal loss for better convergence.
    """
    
    def __init__(self, use_focal=True, focal_alpha=2.0, focal_beta=4.0):
        super(UNetKeypointLoss, self).__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, K, H, W]
            target_heatmaps: Target heatmaps [B, K, H, W]
        """
        if self.use_focal:
            # Focal loss for keypoint detection
            # Helps focus on hard examples
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            
            # Positive focal loss
            pos_loss = -self.focal_alpha * torch.pow(1 - pred_sigmoid, self.focal_beta) * \
                       target_heatmaps * torch.log(pred_sigmoid + 1e-8)
            
            # Negative focal loss
            neg_loss = -(1 - self.focal_alpha) * torch.pow(pred_sigmoid, self.focal_beta) * \
                       (1 - target_heatmaps) * torch.log(1 - pred_sigmoid + 1e-8)
            
            loss = (pos_loss + neg_loss).mean()
        else:
            # Simple MSE loss with sigmoid
            pred_sigmoid = torch.sigmoid(pred_heatmaps)
            loss = self.mse(pred_sigmoid, target_heatmaps)
        
        return loss


class UNetTrainer:
    """Trainer class for UNet model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        save_dir
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
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
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        
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
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs):
        """Train the model for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Save directory: {self.save_dir}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config['training']['learning_rate']
            
            # Print epoch summary
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = self.save_dir / 'best_unet_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': self.config
                }, best_model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint")
        
        # Save final model
        final_model_path = self.save_dir / 'final_unet_model.pth'
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, final_model_path)
        print(f"\n✓ Training complete! Final model saved to {final_model_path}")
        print(f"✓ Best validation loss: {self.best_val_loss:.4f}")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    # Load configuration
    config_path = project_root / 'experiments' / 'configs' / 'unet_config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("UNet Keypoint Detection Training")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Data paths
    data_root = project_root / 'data' / 'Train' / 'Keypointrcnn_data'
    train_image_dir = data_root / 'images' / 'train'
    train_label_dir = data_root / 'labels' / 'train'
    val_image_dir = data_root / 'images' / 'val'
    val_label_dir = data_root / 'labels' / 'val'
    
    print(f"\nData directories:")
    print(f"  Train images: {train_image_dir}")
    print(f"  Train labels: {train_label_dir}")
    print(f"  Val images:   {val_image_dir}")
    print(f"  Val labels:   {val_label_dir}")
    
    # Create preprocessor
    target_size = tuple(config['data']['image_size'])
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=True,
        apply_clahe=True
    )
    
    # Create augmentation for training
    augmentation = SpondylolisthesisAugmentation(mode='train')
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_unet_dataloaders(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        val_image_dir=val_image_dir,
        val_label_dir=val_label_dir,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        heatmap_sigma=3.0,
        output_stride=1,
        preprocessor=preprocessor,
        augmentation=augmentation
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_unet(
        in_channels=3,
        num_keypoints=4,  # 4 corners per vertebra
        bilinear=False,
        base_channels=64
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = UNetKeypointLoss(use_focal=True, focal_alpha=2.0, focal_beta=4.0)
    
    # Optimizer
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler_name = config['training'].get('scheduler', 'StepLR')
    if scheduler_name == 'StepLR':
        step_size = config['training'].get('step_size', 10)
        gamma = config['training'].get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        scheduler = None
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = project_root / 'experiments' / 'results' / f'unet_{timestamp}'
    
    # Create trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        save_dir=save_dir
    )
    
    # Train
    num_epochs = config['training']['num_epochs']
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
