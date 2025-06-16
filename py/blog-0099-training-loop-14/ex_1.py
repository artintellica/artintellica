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

W, B = torch.meshgrid(w_range, b_range, indexing='ij')
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

# Example solution
path_new = gradient_descent_steps(x, y, w_start=0.0, b_start=-1.0)
ws_new, bs_new = zip(*path_new)

plt.figure(figsize=(8, 6))
plt.contourf(W.numpy(), B.numpy(), loss_surface.numpy(), levels=50, cmap='magma')
plt.plot(ws_new, bs_new, marker='o', color='limegreen', label='Gradient Descent Path')
plt.scatter(ws_new[0], bs_new[0], color='white', edgecolor='black', s=100, label='Start')
plt.scatter(ws_new[-1], bs_new[-1], color='orange', edgecolor='black', s=100, label='End')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.title('GD Path with New Starting Point')
plt.legend()
plt.show()
