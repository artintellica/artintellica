# Stochastic Gradient Descent for Linear Regression in PyTorch
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List


# Generate dummy regression data
def generate_data(n: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    X = torch.linspace(0, 1, n).unsqueeze(1)
    y = 2 * X + 1 + 0.1 * torch.randn_like(X)  # y = 2x + 1 + noise
    return X, y


def train_stochastic(
    X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.1
) -> List[float]:
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    n = X.size(0)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in perm:
            xi = X[i]
            yi = y[i]
            # Forward
            y_pred = xi * w + b
            loss = (y_pred - yi) ** 2
            epoch_loss += loss.item()

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
losses_sgd = train_stochastic(X, y, epochs=50, lr=0.05)

plt.plot(losses_sgd, label="Stochastic Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stochastic Gradient Descent Loss Curve")
plt.legend()
plt.show()
