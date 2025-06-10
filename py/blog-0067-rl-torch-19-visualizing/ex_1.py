import torch
import matplotlib.pyplot as plt
import math

# ### **Exercise 1:** Load a Simple 2D Dataset (or Generate Synthetic Data)

# - Generate 300 two-dimensional data points from a normal distribution (mean =
#   [1, 4], covariance = [[2, 1], [1, 3]]).
# - Plot the raw data.
torch.manual_seed(42)
N = 300
mean = torch.tensor([1.0, 4.0])
cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
L = torch.linalg.cholesky(cov)
data = torch.randn(N, 2) @ L.T + mean
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.title("Generated 2D Data")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
