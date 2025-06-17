from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Create a simple regression dataset
class ToyDataset(Dataset):
    def __init__(self, n_samples: int = 100):
        self.x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
        self.y = 3 * self.x + 1 + 0.5 * torch.randn_like(self.x)  # Linear + noise

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

# Create the dataset and dataloaders
dataset = ToyDataset()
batch_size = 16

loader_batch = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loader_single = DataLoader(dataset, batch_size=1, shuffle=True)

