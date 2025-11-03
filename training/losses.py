import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        return nn.CrossEntropyLoss()(predictions, targets)

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = MSELoss()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.alpha * mse + (1 - self.alpha) * ce