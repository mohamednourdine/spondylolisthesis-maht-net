"""Debug version of setup to find where it hangs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("STEP 1: Imports...")
import torch
import numpy as np
from config.unet_config import UNetConfig
from models.model_registry import ModelRegistry

print("STEP 2: Load config...")
config = UNetConfig()
config.BATCH_SIZE = 2
config.NUM_EPOCHS = 1
print(f"  Batch size: {config.BATCH_SIZE}")

print("STEP 3: Set device...")
device = torch.device('cpu')  # Force CPU for testing
print(f"  Device: {device}")

print("STEP 4: Create dataloaders...")
from src.data.unet_dataset import create_unet_dataloaders
print("  Creating train/val loaders...")
train_loader, val_loader = create_unet_dataloaders(
    train_image_dir=config.TRAIN_IMAGE_DIR,
    train_label_dir=config.TRAIN_LABEL_DIR,
    val_image_dir=config.VAL_IMAGE_DIR,
    val_label_dir=config.VAL_LABEL_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=0,  # Single threaded
    heatmap_sigma=config.HEATMAP_SIGMA,
    output_stride=config.OUTPUT_STRIDE
)
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

print("STEP 5: Create model...")
model = ModelRegistry.create('unet', **config.get_model_config())
model = model.to(device)
print(f"  Model created: {model.__class__.__name__}")

print("STEP 6: Create criterion...")
from training.losses import UNetKeypointLoss
criterion = UNetKeypointLoss(
    use_focal=config.USE_FOCAL_LOSS,
    focal_alpha=config.FOCAL_ALPHA,
    focal_beta=config.FOCAL_BETA
)
print(f"  Criterion: {criterion.__class__.__name__}")

print("STEP 7: Create optimizer...")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)
print(f"  Optimizer: Adam")

print("STEP 8: Create scheduler...")
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.STEP_SIZE,
    gamma=config.GAMMA
)
print(f"  Scheduler: StepLR")

print("STEP 9: Create trainer...")
from training.unet_trainer import UNetTrainer
save_dir = config.RESULTS_DIR
experiment_name = 'debug_test'

trainer = UNetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    config=config.to_dict(),
    save_dir=save_dir,
    scheduler=scheduler,
    experiment_name=experiment_name,
    compute_metrics=True
)
print(f"  Trainer created!")
print(f"  Checkpoint dir: {trainer.checkpoint_dir}")

print("\nSTEP 10: All setup complete!")
print("Ready to call trainer.train()")
