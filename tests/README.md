# Tests Directory

This directory contains all test scripts for the spondylolisthesis detection project.

## Test Scripts

### Component Tests
- `test_unet.py` - Comprehensive UNet model tests (architecture, forward pass, dataset, training)
- `test_training_system.py` - Tests for training system components (config, registry, trainer)
- `test_global_evaluator.py` - Tests for the global keypoint evaluator

### Integration Tests
- `test_training_small.py` - Full training test with small data subset (10 samples)
- `test_quick_training.py` - Quick training components test
- `test_debug_setup.py` - Debug setup for training pipeline

### System Tests
- `test_train_imports.py` - Test train.py imports
- `test_train_help.py` - Test train.py command line interface

## Running Tests

Run individual tests:
```bash
python tests/test_unet.py
python tests/test_training_small.py
```

Run all tests (if using pytest):
```bash
pytest tests/
```

## Quick Training Test

To verify everything works with a small dataset:
```bash
cd /Users/mnourdine/phd/spondylolisthesis-maht-net
conda activate phd
python tests/test_training_small.py
```

This will train for 2 epochs on just 10 samples and verify all metrics are computed correctly.
