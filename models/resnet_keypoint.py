from torchvision import models
import torch
import torch.nn as nn

class ResNetKeypointDetector(nn.Module):
    def __init__(self, num_keypoints):
        super(ResNetKeypointDetector, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the final layer to output keypoints
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_keypoints * 2)  # x and y coordinates for each keypoint

    def forward(self, x):
        # Forward pass through the ResNet model
        x = self.resnet(x)
        # Reshape output to (batch_size, num_keypoints, 2)
        return x.view(-1, num_keypoints, 2)