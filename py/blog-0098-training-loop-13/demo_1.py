# Batch Gradient Descent for Linear Regression in PyTorch
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


X, y = generate_data()
losses_batch = train_batch(X, y, epochs=50, lr=0.5)

plt.plot(losses_batch, label="Batch Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Batch Gradient Descent Loss Curve")
plt.legend()
plt.show()
