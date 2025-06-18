import torch

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA GPU!")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple M1/M2 GPU (MPS)!")
        return torch.device("mps")
    else:
        print("Using CPU.")
        return torch.device("cpu")

import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

from torch.utils.data import Dataset, DataLoader
import torch

class RandomDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int) -> None:
        self.X = torch.randn(n_samples, input_dim)
        self.y = (self.X.sum(dim=1, keepdim=True) > 0).float()  # Simple binary target

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

import torch.optim as optim

def train_parallel(
    n_gpus: int = 2,  # Number of GPUs to parallelize over (will auto-use available)
    epochs: int = 5,
    batch_size: int = 128
) -> None:
    device = get_best_device()
    dataset = RandomDataset(10_000, input_dim=20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_dim=20, hidden_dim=64, output_dim=1)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_parallel()
