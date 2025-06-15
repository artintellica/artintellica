import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_data(n_samples: int = 100) -> Tuple[Tensor, Tensor]:
    """Generate synthetic linear data with noise."""
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)  # shape: (n_samples, 1)
    true_w, true_b = 2.0, -1.0
    y = true_w * X + true_b + 0.5 * torch.randn_like(X)
    return X, y


def train_linear_regression(
    X: Tensor, y: Tensor, n_epochs: int = 200, lr: float = 0.05
) -> Tuple[Tensor, Tensor, List[float]]:
    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)
    losses = []

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X * w + b  # shape: (n_samples, 1)
        loss = torch.mean((y_pred - y) ** 2)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # (Optional) Print progress
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}"
            )

    return w.detach(), b.detach(), losses


def plot_results(
    X: Tensor, y: Tensor, w: Tensor, b: Tensor, losses: List[float]
) -> None:
    plt.figure(figsize=(12, 5))

    # Plot data and fitted line
    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), label="Data")
    plt.plot(X.numpy(), (X * w + b).numpy(), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")

    # Plot loss over time
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate data
    X, y = generate_data()
    # Train the model
    w, b, losses = train_linear_regression(X, y)
    # Visualize the results
    plot_results(X, y, w, b, losses)
