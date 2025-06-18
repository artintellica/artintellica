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
