import math, random, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print("Using", device)


class MLP(nn.Module):
    """Three‑layer tanh MLP for 1‑D regression."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


def gen_batch(n: int):
    """x ∈ [‑π, π],  y = sin x ."""
    x = torch.rand(n, 1) * (2 * math.pi) - math.pi
    y = torch.sin(x)
    return x.to(device), y.to(device)


@torch.no_grad()
def mse(model, n_val=1_000):
    x_val, y_val = gen_batch(n_val)
    return nn.functional.mse_loss(model(x_val), y_val).item()


def train_once(width: int, n_train: int, epochs: int = 500, lr: float = 1e-2):
    model = MLP(width).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    x_train, y_train = gen_batch(n_train)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        opt.step()
    return mse(model)
