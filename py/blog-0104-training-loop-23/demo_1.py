import torch
from torch import Tensor
import matplotlib.pyplot as plt


def generate_sine_data(n_samples: int = 100) -> tuple[Tensor, Tensor]:
    # Inputs: shape (N, 1)
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n_samples).unsqueeze(1)
    # Outputs: shape (N, 1)
    y = torch.sin(x)
    return x, y


X, y_true = generate_sine_data(300)

D_in = 1  # Input size
D_hidden = 10  # Hidden layer width
D_out = 1  # Output size

# Manual parameter initialization with requires_grad=True
W1 = torch.randn(D_in, D_hidden, requires_grad=True)
b1 = torch.zeros(D_hidden, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)

learning_rate = 1e-2
n_epochs = 1000

losses: list[float] = []

for epoch in range(n_epochs):
    # Forward pass: linear, activation, linear
    h: Tensor = X @ W1 + b1  # shape: (N, D_hidden)
    h_relu: Tensor = torch.relu(h)
    y_pred: Tensor = h_relu @ W2 + b2  # shape: (N, D_out)

    # Compute and record loss
    loss: Tensor = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())  # type: ignore

    # Backward pass
    loss.backward()

    # Update parameters using SGD (gradient descent)
    with torch.no_grad():
        W1 -= learning_rate * W1.grad if W1.grad is not None else 0
        b1 -= learning_rate * b1.grad if b1.grad is not None else 0
        W2 -= learning_rate * W2.grad if W2.grad is not None else 0
        b2 -= learning_rate * b2.grad if b2.grad is not None else 0

        # Zero gradients for next iteration
        W1.grad.zero_() if W1.grad is not None else None
        b1.grad.zero_() if b1.grad is not None else None
        W2.grad.zero_() if W2.grad is not None else None
        b2.grad.zero_() if b2.grad is not None else None

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}, Loss: {loss.item():.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

# Model predictions
with torch.no_grad():
    h = X @ W1 + b1
    h_relu = torch.relu(h)
    y_pred = h_relu @ W2 + b2

plt.plot(X.numpy(), y_true.numpy(), label="True Function")
plt.plot(X.numpy(), y_pred.numpy(), label="Neural Net Prediction")
plt.legend()
plt.title("Neural Network Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot training loss
plt.figure()
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
