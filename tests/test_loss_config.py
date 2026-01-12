"""
Quick test to verify the updated loss function configuration works.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.mac_config import MacConfig
from training.losses import (
    UNetKeypointLoss, 
    AdaptiveWingLoss, 
    CombinedKeypointLoss,
    MSEWithWeightedBackground
)
import torch

def test_loss_function_creation():
    """Test that all loss functions can be created from config."""
    
    print("="*60)
    print("Testing Loss Function Configuration")
    print("="*60)
    
    config = MacConfig
    
    # Test all loss function options
    loss_options = ['adaptive_wing', 'combined', 'weighted_mse', 'focal']
    
    for loss_name in loss_options:
        print(f"\nTesting: {loss_name}")
        
        # Create criterion based on loss name
        if loss_name == 'adaptive_wing':
            criterion = AdaptiveWingLoss(
                omega=config.AWING_OMEGA,
                theta=config.AWING_THETA,
                epsilon=config.AWING_EPSILON,
                alpha=config.AWING_ALPHA
            )
        elif loss_name == 'combined':
            criterion = CombinedKeypointLoss()
        elif loss_name == 'weighted_mse':
            criterion = MSEWithWeightedBackground()
        elif loss_name == 'focal':
            criterion = UNetKeypointLoss(
                use_focal=True,
                focal_alpha=config.FOCAL_ALPHA,
                focal_beta=config.FOCAL_BETA
            )
        
        # Test with dummy data
        pred = torch.randn(2, 4, 128, 128)
        target = torch.rand(2, 4, 128, 128)
        
        try:
            loss = criterion(pred, target)
            print(f"  ✓ {loss_name}: Loss = {loss.item():.4f}")
        except Exception as e:
            print(f"  ✗ {loss_name}: ERROR - {e}")
            return False
    
    print("\n" + "="*60)
    print(f"Current config loss function: {config.LOSS_FUNCTION}")
    print("✓ All loss functions working correctly!")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = test_loss_function_creation()
    exit(0 if success else 1)
