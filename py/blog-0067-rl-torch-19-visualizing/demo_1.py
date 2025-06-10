import torch
import matplotlib.pyplot as plt

# Generate 2D Gaussian blobs
torch.manual_seed(42)
N = 200
mean = torch.tensor([2.0, -3.0])
cov = torch.tensor([[3.0, 1.2], [1.2, 2.0]])
L = torch.linalg.cholesky(cov)
data = torch.randn(N, 2) @ L.T + mean

# Visualize original data
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.title("Original 2D Data")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
