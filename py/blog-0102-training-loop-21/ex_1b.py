from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch

def generate_moons(n_samples: int = 200, noise: float = 0.20) -> Tuple[torch.Tensor, torch.Tensor]:
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y

X, y = generate_moons()
# plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", s=40)
# plt.title("Synthetic Nonlinear Dataset: Two Moons")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))  # For binary classification

model = LinearModel(input_dim=2, output_dim=1)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop (very simple!)
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")

import numpy as np

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), 
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)
    plt.contourf(xx, yy, probs, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", s=40)
    plt.title("Linear Model: Can Capture Nonlinearity")
    plt.show()

# plot_decision_boundary(model, X, y)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Sigmoid()  # Try swapping to nn.Tanh() or nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return torch.sigmoid(self.fc2(h))  # For binary classification

mlp = MLP(input_dim=2, hidden_dim=8, output_dim=1)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = mlp(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")
plot_decision_boundary(mlp, X, y)
