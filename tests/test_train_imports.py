"""Test train.py import."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing train.py imports...")
print("1. Importing train module...")
import train
print("✓ Import successful")

print("\n2. Checking ModelRegistry...")
from models.model_registry import ModelRegistry
print(f"   Available models: {ModelRegistry.list_models()}")
print("✓ Registry works")

print("\n3. Checking config loading...")
from config.unet_config import UNetConfig
config = UNetConfig()
print(f"   Config loaded: {config.MODEL_NAME}")
print(f"   Batch size: {config.BATCH_SIZE}")
print("✓ Config works")

print("\n✓ All imports successful!")
