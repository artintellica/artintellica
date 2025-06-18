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

import torch.nn as nn


def train(
    loader: DataLoader, model: nn.Module, lr: float = 0.1, epochs: int = 20
) -> list:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        batch_losses = []
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(sum(batch_losses) / len(batch_losses))
    return losses


# # Models
# model_single = nn.Linear(1, 1)
# model_batch = nn.Linear(1, 1)

# # Train
# losses_single = train(loader_single, model_single)
# losses_batch = train(loader_batch, model_batch)

# # Plot
# plt.plot(losses_single, label="Batch size = 1 (SGD)")
# plt.plot(losses_batch, label="Batch size = 16")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training Loss: Single Sample vs. Mini-batch")
# plt.show()

batch_sizes = [1, 8, 32, 64]
losses_all = []

for b in batch_sizes:
    loader = DataLoader(dataset, batch_size=b, shuffle=True)
    model = nn.Linear(1, 1)
    losses = train(loader, model)
    losses_all.append(losses)

for i, losses in enumerate(losses_all):
    plt.plot(losses, label=f'Batch size = {batch_sizes[i]}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Effect of Batch Size on Training Loss')
plt.show()
