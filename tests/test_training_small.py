"""Test training with a small subset of data."""

import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("Testing Training with Small Data Subset")
print("="*60 + "\n")

# Import required modules
from config.unet_config import UNetConfig
from models.model_registry import ModelRegistry
from src.data.unet_dataset import UNetSpondylolisthesisDataset, unet_collate_fn
from training.losses import UNetKeypointLoss
from training.unet_trainer import UNetTrainer
from torch.utils.data import DataLoader, Subset

# Configure
config = UNetConfig()
config.BATCH_SIZE = 2
config.NUM_EPOCHS = 2
device = torch.device('cpu')

print("1. Loading datasets...")
# Load full datasets
train_dataset = UNetSpondylolisthesisDataset(
    image_dir=config.TRAIN_IMAGE_DIR,
    label_dir=config.TRAIN_LABEL_DIR,
    mode='train',
    heatmap_sigma=config.HEATMAP_SIGMA,
    output_stride=config.OUTPUT_STRIDE
)

val_dataset = UNetSpondylolisthesisDataset(
    image_dir=config.VAL_IMAGE_DIR,
    label_dir=config.VAL_LABEL_DIR,
    mode='val',
    heatmap_sigma=config.HEATMAP_SIGMA,
    output_stride=config.OUTPUT_STRIDE
)

print(f"   Full train dataset: {len(train_dataset)} samples")
print(f"   Full val dataset: {len(val_dataset)} samples")

# Create small subsets - just 10 samples each for quick testing
train_subset = Subset(train_dataset, list(range(min(10, len(train_dataset)))))
val_subset = Subset(val_dataset, list(range(min(10, len(val_dataset)))))

print(f"   Using train subset: {len(train_subset)} samples")
print(f"   Using val subset: {len(val_subset)} samples\n")

# Create dataloaders
train_loader = DataLoader(
    train_subset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=unet_collate_fn
)

val_loader = DataLoader(
    val_subset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=unet_collate_fn
)

print(f"2. Creating model...")
model = ModelRegistry.create('unet', **config.get_model_config())
model = model.to(device)
print(f"   Model: {model.__class__.__name__}")
print(f"   Device: {device}\n")

print("3. Setting up training components...")
criterion = UNetKeypointLoss(
    use_focal=config.USE_FOCAL_LOSS,
    focal_alpha=config.FOCAL_ALPHA,
    focal_beta=config.FOCAL_BETA
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.STEP_SIZE,
    gamma=config.GAMMA
)

print(f"   Criterion: {criterion.__class__.__name__}")
print(f"   Optimizer: Adam (lr={config.LEARNING_RATE})")
print(f"   Scheduler: StepLR\n")

print("4. Creating trainer...")
config_dict = config.to_dict()
config_dict['model_name'] = 'unet'  # Add model name for better folder organization

trainer = UNetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    config=config_dict,
    save_dir=config.RESULTS_DIR,
    scheduler=scheduler,
    experiment_name='small_data_test',
    compute_metrics=True
)
print(f"   Trainer created!")
print(f"   Save directory: {trainer.save_dir}\n")

print("="*60)
print("Starting Training (2 epochs, 10 train + 10 val samples)")
print("="*60 + "\n")

# Train
trainer.train(num_epochs=config.NUM_EPOCHS)

print("\n" + "="*60)
print("✓ Training Test Complete!")
print("="*60)
print("\nKey observations:")
print(f"  - Trained on {len(train_subset)} samples")
print(f"  - Validated on {len(val_subset)} samples")
print(f"  - Metrics tracked: MRE, SDR (2mm, 2.5mm, 3mm, 4mm)")
print(f"  - Model saved to: {trainer.save_dir}")
print("\nTo train on full dataset in the cloud:")
print("  python train.py --model unet --epochs 50 --batch-size 16")
