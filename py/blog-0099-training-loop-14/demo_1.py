import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <-- Add this line

# Set seed for reproducibility
torch.manual_seed(42)


# Generate data
def create_data(n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-3, 3, n_samples)
    y = 2 * x + 1 + torch.randn(n_samples) * 0.5
    return x, y


x, y = create_data()

# Set grid ranges
w_range = torch.linspace(0.0, 4.0, 100)
b_range = torch.linspace(-1.0, 3.0, 100)

W, B = torch.meshgrid(w_range, b_range, indexing="ij")
loss_surface = torch.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(B.shape[1]):
        w, b = W[i, j], B[i, j]
        y_pred = w * x + b
        loss = torch.mean((y - y_pred) ** 2)
        loss_surface[i, j] = loss

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W.numpy(), B.numpy(), loss_surface.numpy(), cmap='viridis', alpha=0.8) # type: ignore
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss') # type: ignore
ax.set_title('Loss Surface of Linear Regression')
plt.show()

def gradient_descent_steps(
    x: torch.Tensor,
    y: torch.Tensor,
    w_start: float = 3.5,
    b_start: float = 2.5,
    lr: float = 0.1,
    steps: int = 15,
) -> list[tuple[float, float]]:
    w = torch.tensor(w_start, requires_grad=True)
    b = torch.tensor(b_start, requires_grad=True)
    path = [(w.item(), b.item())]
    for _ in range(steps):
        y_pred = w * x + b
        loss = torch.mean((y - y_pred) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad if w.grad is not None else 0
            b -= lr * b.grad if b.grad is not None else 0
        w.grad.zero_() if w.grad is not None else None
        b.grad.zero_() if b.grad is not None else None
        path.append((w.item(), b.item()))
    return path


# Run optimization
path = gradient_descent_steps(x, y)
ws, bs = zip(*path)

# Overlay the path on contour plot
plt.figure(figsize=(8, 6))
plt.contourf(W.numpy(), B.numpy(), loss_surface.numpy(), levels=50, cmap="viridis")
plt.plot(ws, bs, marker="o", color="red", label="Gradient Descent Path")
plt.scatter(ws[0], bs[0], color="white", edgecolor="black", s=100, label="Start")
plt.scatter(ws[-1], bs[-1], color="yellow", edgecolor="black", s=100, label="End")
plt.xlabel("Weight (w)")
plt.ylabel("Bias (b)")
plt.title("Gradient Descent Path on the Loss Landscape")
plt.legend()
plt.show()
