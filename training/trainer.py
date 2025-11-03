from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.dataset import SpondylolisthesisDataset
from src.models.maht_net import MAHTNet  # Import your model here
from src.training.losses import CustomLoss  # Import your loss function here

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 20

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data loading
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SpondylolisthesisDataset(root='data/processed/train', transform=train_transforms)
    val_dataset = SpondylolisthesisDataset(root='data/processed/val', transform=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = MAHTNet().to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()