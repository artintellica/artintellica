# Mini-batch Gradient Descent in PyTorch
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List


# Generate dummy regression data
def generate_data(n: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    X = torch.linspace(0, 1, n).unsqueeze(1)
    y = 2 * X + 1 + 0.1 * torch.randn_like(X)  # y = 2x + 1 + noise
    return X, y


# Linear regression batch update
def train_batch(
    X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.1
) -> List[float]:
    # Initialize weights and bias
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    losses = []

    for epoch in range(epochs):
        # Forward
        y_pred = X * w + b
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            w -= lr * w.grad if w.grad is not None else 0
            b -= lr * b.grad if b.grad is not None else 0

        # Zero gradients
        w.grad.zero_() if w.grad is not None else None
        b.grad.zero_() if b.grad is not None else None

    return losses


def train_minibatch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 0.1,
) -> List[float]:
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    n = X.size(0)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xi = X[idx]
            yi = y[idx]
            # Forward
            y_pred = xi * w + b
            loss = ((y_pred - yi) ** 2).mean()
            epoch_loss += loss.item() * xi.size(0)

            # Backward
            loss.backward()

            # Update
            with torch.no_grad():
                w -= lr * w.grad if w.grad is not None else 0
                b -= lr * b.grad if b.grad is not None else 0

            w.grad.zero_() if w.grad is not None else None
            b.grad.zero_() if b.grad is not None else None
        losses.append(epoch_loss / n)
    return losses


X, y = generate_data()
# losses_minibatch = train_minibatch(X, y, batch_size=16, epochs=50, lr=0.1)

X_large, y_large = generate_data(10_000)

import time

start = time.time()
losses_large_batch = train_batch(X_large, y_large, epochs=10, lr=0.5)
print("Batch GD Time:", time.time() - start)

start = time.time()
losses_large_mb = train_minibatch(X_large, y_large, batch_size=64, epochs=10, lr=0.5)
print("Mini-batch GD Time:", time.time() - start)
