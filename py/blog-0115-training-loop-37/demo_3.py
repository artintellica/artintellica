
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

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.ToTensor()

import torch.nn as nn
import torch.nn.functional as F

class MLPWithNormalization(nn.Module):
    def __init__(self, use_batchnorm: bool = False, use_layernorm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
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
