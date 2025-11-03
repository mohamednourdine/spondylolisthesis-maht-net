from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

class KeypointRCNNModel:
    def __init__(self, num_classes, num_keypoints):
        # Define the anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        # Define the Keypoint R-CNN model
        self.model = KeypointRCNN(
            backbone=self._get_backbone(),
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            rpn_anchor_generator=self.anchor_generator
        )

    def _get_backbone(self):
        # Load a pre-trained ResNet model and return the backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        return backbone

    def transform(self, image):
        # Define the transformations for the input image
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def forward(self, images):
        # Forward pass through the model
        return self.model(images)