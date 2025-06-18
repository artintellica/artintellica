import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")

import torchvision.transforms as transforms

train_transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.ToTensor()

import torch.nn as nn
import torch.nn.functional as F


class MLPWithNormalization(nn.Module):
    def __init__(self, use_batchnorm: bool = False, use_layernorm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        if use_batchnorm:
            self.norm1 = nn.BatchNorm1d(128)
        elif use_layernorm:
            self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_batchnorm or self.use_layernorm:
            x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Stops training if validation loss doesn't improve after 'patience' epochs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 1e-3,
    use_early_stopping: bool = True,
    patience: int = 5,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                val_loss += criterion(y_pred, yb).item()
                preds = y_pred.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if use_early_stopping and early_stopper(val_loss):
            print(f"No improvement after {patience} epochs. Stopping early.")
            break


# Download and prepare datasets
train_dataset = MNIST(root=".", train=True, download=True, transform=train_transform)
val_dataset = MNIST(root=".", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Try batch norm
model_bn = MLPWithNormalization(use_batchnorm=True)
train(model_bn, train_loader, val_loader, device)

# Try layer norm
model_ln = MLPWithNormalization(use_layernorm=True)
train(model_ln, train_loader, val_loader, device)

import matplotlib.pyplot as plt


def show_batch(dataset, n: int = 8) -> None:
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))
    plt.figure(figsize=(12, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        plt.axis("off")
        plt.title(str(labels[i].item()))
    plt.show()


show_batch(train_dataset, n=8)
