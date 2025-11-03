class MAHTNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MAHTNet, self).__init__()
        # Define the layers of the MAHT-Net model here
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.classifier = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def build_encoder(self):
        # Define the encoder part of the network
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def build_decoder(self):
        # Define the decoder part of the network
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

    def compute_loss(self, predictions, targets):
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, targets)