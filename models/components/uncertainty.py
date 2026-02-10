"""
Uncertainty Estimation for MAHT-Net

Provides multiple methods for estimating prediction uncertainty:
1. Heatmap-based: Derive uncertainty from heatmap spread
2. Learned: Dedicated head that predicts σ_x, σ_y directly
3. MC Dropout: Multiple forward passes with dropout (inference-time)

Clinical use: Flag predictions with high uncertainty for manual review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class UncertaintyHead(nn.Module):
    """
    Learned uncertainty estimation head.
    
    Predicts uncertainty (σ_x, σ_y) for each keypoint directly
    from the feature maps or VAM output.
    
    Args:
        d_model: Input feature dimension (256)
        num_keypoints: Number of keypoints (20 for AP, 22 for LA)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        min_sigma: Minimum uncertainty (prevents collapse)
        max_sigma: Maximum uncertainty (prevents explosion)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_keypoints: int = 20,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_sigma: float = 0.5,
        max_sigma: float = 50.0
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        # MLP for uncertainty prediction
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # σ_x, σ_y
        )
        
        # Initialize to predict moderate uncertainty
        self._init_weights()
        
        print(f"  ✓ UncertaintyHead: {num_keypoints} keypoints")
    
    def _init_weights(self):
        """Initialize to output moderate uncertainty."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize last layer to output ~2.0 sigma
        with torch.no_grad():
            self.uncertainty_mlp[-1].bias.fill_(0.7)  # exp(0.7) ≈ 2.0
    
    def forward(self, vam_features: torch.Tensor) -> torch.Tensor:
        """
        Predict uncertainty from VAM features.
        
        Args:
            vam_features: (B, K, d_model) features from VAM
            
        Returns:
            sigma: (B, K, 2) uncertainty (σ_x, σ_y) per keypoint
        """
        B, K, C = vam_features.shape
        
        # Predict log-sigma for each keypoint
        log_sigma = self.uncertainty_mlp(vam_features)  # (B, K, 2)
        
        # Convert to sigma and clamp
        sigma = torch.exp(log_sigma)
        sigma = sigma.clamp(self.min_sigma, self.max_sigma)
        
        return sigma


class HeatmapUncertaintyHead(nn.Module):
    """
    CNN-based uncertainty estimation from heatmaps.
    
    Takes the predicted heatmaps and extracts uncertainty
    using spatial statistics.
    
    Args:
        num_keypoints: Number of keypoints
        hidden_channels: Hidden conv channels
    """
    
    def __init__(
        self,
        num_keypoints: int = 20,
        hidden_channels: int = 64
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Per-keypoint conv for local uncertainty estimation
        self.conv = nn.Sequential(
            nn.Conv2d(num_keypoints, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(hidden_channels, num_keypoints * 2)
        
        print(f"  ✓ HeatmapUncertaintyHead: {num_keypoints} keypoints")
    
    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty from predicted heatmaps.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            
        Returns:
            sigma: (B, K, 2) uncertainty (σ_x, σ_y)
        """
        B, K, H, W = heatmaps.shape
        
        # Apply conv to heatmaps
        features = self.conv(heatmaps)  # (B, hidden, 1, 1)
        features = features.view(B, -1)  # (B, hidden)
        
        # Predict sigma
        log_sigma = self.fc(features)  # (B, K*2)
        log_sigma = log_sigma.view(B, K, 2)
        
        sigma = F.softplus(log_sigma) + 0.5  # Ensure positive, min 0.5
        
        return sigma


class HeatmapSpreadUncertainty(nn.Module):
    """
    Extract uncertainty directly from heatmap spread.
    
    No learned parameters - directly computes spatial variance
    of the heatmap around its peak.
    
    This is the simplest and most interpretable approach.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from heatmap spatial variance.
        
        Args:
            heatmaps: (B, K, H, W) predicted heatmaps
            
        Returns:
            sigma: (B, K, 2) uncertainty (σ_x, σ_y) in pixels
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        dtype = heatmaps.dtype
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=dtype, device=device)
        x_coords = torch.arange(W, dtype=dtype, device=device)
        
        # Convert heatmaps to probability with temperature
        heatmaps_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(heatmaps_flat / self.temperature, dim=-1)
        probs = probs.view(B, K, H, W)
        
        # Compute expected coordinates (mean)
        # E[x] = sum_y(sum_x(p(x,y) * x))
        x_probs = probs.sum(dim=2)  # (B, K, W)
        y_probs = probs.sum(dim=3)  # (B, K, H)
        
        x_mean = (x_probs * x_coords).sum(dim=-1)  # (B, K)
        y_mean = (y_probs * y_coords).sum(dim=-1)  # (B, K)
        
        # Compute variance
        # Var[x] = E[x²] - E[x]²
        x_sq_mean = (x_probs * x_coords**2).sum(dim=-1)
        y_sq_mean = (y_probs * y_coords**2).sum(dim=-1)
        
        x_var = x_sq_mean - x_mean**2
        y_var = y_sq_mean - y_mean**2
        
        # Standard deviation (ensure positive)
        sigma_x = torch.sqrt(F.relu(x_var) + 1e-8)
        sigma_y = torch.sqrt(F.relu(y_var) + 1e-8)
        
        sigma = torch.stack([sigma_x, sigma_y], dim=-1)  # (B, K, 2)
        
        return sigma


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled
    and computes prediction variance.
    
    Note: This is used at inference time and requires
    a model with dropout layers.
    
    Args:
        num_samples: Number of MC forward passes
    """
    
    def __init__(self, num_samples: int = 10):
        super().__init__()
        self.num_samples = num_samples
    
    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using MC Dropout.
        
        Args:
            model: MAHT-Net model (with dropout layers)
            x: (B, 3, H, W) input images
            
        Returns:
            mean_coords: (B, K, 2) mean predicted coordinates
            sigma: (B, K, 2) standard deviation (uncertainty)
            mean_heatmaps: (B, K, H, W) mean heatmaps
        """
        # Enable dropout during inference
        model.train()
        
        all_heatmaps = []
        
        for _ in range(self.num_samples):
            output = model(x)
            all_heatmaps.append(output['heatmaps'])
        
        # Stack and compute statistics
        all_heatmaps = torch.stack(all_heatmaps, dim=0)  # (S, B, K, H, W)
        mean_heatmaps = all_heatmaps.mean(dim=0)
        
        # Extract coordinates from each sample
        from src.utils.dark_decoding import soft_argmax
        
        all_coords = []
        for i in range(self.num_samples):
            coords = soft_argmax(all_heatmaps[i])
            all_coords.append(coords)
        
        all_coords = torch.stack(all_coords, dim=0)  # (S, B, K, 2)
        mean_coords = all_coords.mean(dim=0)
        sigma = all_coords.std(dim=0)
        
        model.eval()
        
        return mean_coords, sigma, mean_heatmaps


class CombinedUncertainty(nn.Module):
    """
    Combines multiple uncertainty estimation methods.
    
    Uses both learned and heatmap-based uncertainty,
    allowing the model to leverage complementary signals.
    
    Args:
        d_model: Feature dimension
        num_keypoints: Number of keypoints
        use_learned: Include learned uncertainty head
        use_heatmap: Include heatmap spread uncertainty
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_keypoints: int = 20,
        use_learned: bool = True,
        use_heatmap: bool = True
    ):
        super().__init__()
        
        self.use_learned = use_learned
        self.use_heatmap = use_heatmap
        
        if use_learned:
            self.learned_head = UncertaintyHead(d_model, num_keypoints)
        
        if use_heatmap:
            self.heatmap_uncertainty = HeatmapSpreadUncertainty()
        
        # Combine weights (learned)
        if use_learned and use_heatmap:
            self.weight_learned = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        vam_features: Optional[torch.Tensor] = None,
        heatmaps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined uncertainty.
        
        Args:
            vam_features: (B, K, d_model) VAM output features
            heatmaps: (B, K, H, W) predicted heatmaps
            
        Returns:
            sigma: (B, K, 2) combined uncertainty
        """
        sigmas = []
        
        if self.use_learned and vam_features is not None:
            sigma_learned = self.learned_head(vam_features)
            sigmas.append(sigma_learned)
        
        if self.use_heatmap and heatmaps is not None:
            sigma_heatmap = self.heatmap_uncertainty(heatmaps)
            sigmas.append(sigma_heatmap)
        
        if len(sigmas) == 0:
            raise ValueError("Must provide vam_features or heatmaps")
        elif len(sigmas) == 1:
            return sigmas[0]
        else:
            # Weighted combination
            w = torch.sigmoid(self.weight_learned)
            return w * sigmas[0] + (1 - w) * sigmas[1]


class NLLLoss(nn.Module):
    """
    Negative Log Likelihood loss for uncertainty-aware training.
    
    Trains the model to predict well-calibrated uncertainties
    by maximizing the likelihood of the true coordinates.
    
    L = 0.5 * log(σ²) + 0.5 * (coord - coord_pred)² / σ²
    
    Args:
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL loss with predicted uncertainty.
        
        Args:
            pred_coords: (B, K, 2) predicted coordinates
            gt_coords: (B, K, 2) ground truth coordinates
            sigma: (B, K, 2) predicted uncertainty
            
        Returns:
            loss: Scalar NLL loss
        """
        # Ensure positive variance
        var = sigma**2 + 1e-8
        
        # Squared error
        sq_error = (pred_coords - gt_coords)**2
        
        # NLL = 0.5 * log(var) + 0.5 * error / var
        nll = 0.5 * torch.log(var) + 0.5 * sq_error / var
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


def test_uncertainty():
    """Test uncertainty estimation modules."""
    print("\n" + "="*60)
    print("Testing Uncertainty Estimation")
    print("="*60)
    
    B, K, H, W = 2, 20, 512, 512
    d_model = 256
    
    # Create dummy data
    vam_features = torch.randn(B, K, d_model)
    heatmaps = torch.randn(B, K, H, W)
    
    # Test learned uncertainty head
    print("\n1. Testing UncertaintyHead...")
    head = UncertaintyHead(d_model, K)
    sigma_learned = head(vam_features)
    print(f"   VAM features: {vam_features.shape}")
    print(f"   Sigma output: {sigma_learned.shape}")
    print(f"   Sigma range: [{sigma_learned.min():.2f}, {sigma_learned.max():.2f}]")
    assert sigma_learned.shape == (B, K, 2)
    print("   ✓ UncertaintyHead passed!")
    
    # Test heatmap spread uncertainty
    print("\n2. Testing HeatmapSpreadUncertainty...")
    spread = HeatmapSpreadUncertainty()
    sigma_spread = spread(heatmaps)
    print(f"   Heatmaps: {heatmaps.shape}")
    print(f"   Sigma output: {sigma_spread.shape}")
    print(f"   Sigma range: [{sigma_spread.min():.2f}, {sigma_spread.max():.2f}]")
    assert sigma_spread.shape == (B, K, 2)
    print("   ✓ HeatmapSpreadUncertainty passed!")
    
    # Test combined uncertainty
    print("\n3. Testing CombinedUncertainty...")
    combined = CombinedUncertainty(d_model, K)
    sigma_combined = combined(vam_features, heatmaps)
    print(f"   Combined sigma: {sigma_combined.shape}")
    assert sigma_combined.shape == (B, K, 2)
    print("   ✓ CombinedUncertainty passed!")
    
    # Test NLL loss
    print("\n4. Testing NLLLoss...")
    nll_loss = NLLLoss()
    pred_coords = torch.randn(B, K, 2) * 50 + 256
    gt_coords = pred_coords + torch.randn(B, K, 2) * 5
    sigma = torch.ones(B, K, 2) * 5.0
    
    loss = nll_loss(pred_coords, gt_coords, sigma)
    print(f"   NLL Loss: {loss.item():.4f}")
    print("   ✓ NLLLoss passed!")
    
    print("\n" + "="*60)
    print("All uncertainty tests passed! ✓")
    print("="*60)
    return True


if __name__ == "__main__":
    test_uncertainty()
