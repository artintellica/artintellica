import torch
import matplotlib.pyplot as plt
from typing import List

# Make reproducible
torch.manual_seed(42)

true_w = 2.0  # True slope
true_b = -1.0  # True intercept

N = 100  # Number of data points

# x: Uniformly spaced between -1 and 1
x: torch.Tensor = torch.linspace(-1, 1, N).unsqueeze(1)  # Shape (N, 1)
# y: Linear relation plus some noise
y: torch.Tensor = true_w * x + true_b + 0.1 * torch.randn(N, 1)

plt.scatter(x.numpy(), y.numpy(), label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fake Linear Data")
plt.legend()
plt.show()

# Initialize parameters (weights and bias) randomly
w: torch.Tensor = torch.randn(1, requires_grad=False)
b: torch.Tensor = torch.randn(1, requires_grad=False)

learning_rate: float = 0.1
num_epochs: int = 50

losses: List[float] = []

for epoch in range(num_epochs):
    # 1. Forward pass: compute predictions
    y_pred: torch.Tensor = w * x + b

    # 2. Compute loss (mean squared error)
    loss: torch.Tensor = ((y_pred - y) ** 2).mean()
    losses.append(loss.item())  # type: ignore

    # 3. Compute gradients (dL/dw and dL/db) using math above
    grad_w: torch.Tensor = (2 / N) * torch.sum((y_pred - y) * x)
    grad_b: torch.Tensor = (2 / N) * torch.sum(y_pred - y)

    # 4. Update parameters manually
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:2d}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}"
        )

print(f"\nLearned parameters: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"True parameters:    w = {true_w:.4f}, b = {true_b:.4f}")

# Plot data and learned regression line
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), (w * x + b).detach().numpy(), color='red', label='Learned Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit (PyTorch)')
plt.show()

# Plot loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss During Training')
plt.show()
