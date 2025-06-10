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

# ### **Exercise 4:** Center and Normalize the Data with PyTorch

# - Subtract the mean from all points (centering).
# - Divide by the standard deviation along each axis (normalizing).
# - Show the scatterplot after normalization and check that variance is $1$ in
#   each direction.

mean_vec: torch.Tensor = data.mean(dim=0)
data_centered: torch.Tensor = data - mean_vec
std_dev: torch.Tensor = data_centered.std(dim=0)
data_normalized: torch.Tensor = data_centered / std_dev
# Visualize normalized data
plt.scatter(data_normalized[:, 0], data_normalized[:, 1], alpha=0.7)
plt.title("Centered & Normalized Data (Unit Std)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()

