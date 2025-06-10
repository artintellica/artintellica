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
# ### **Exercise 3:** Compute the Mean and Covariance Matrix of the Dataset

# - Calculate and print the mean and covariance of your dataset using torch
#   operations.
# - Interpret the covariance matrix (are features correlated?).
mean_vec: torch.Tensor = data.mean(dim=0)
print("Mean vector:\n", mean_vec)
# Centered data
data_centered: torch.Tensor = data - mean_vec
# Covariance matrix
cov_matrix: torch.Tensor = (data_centered.T @ data_centered) / data.shape[0]
print("Covariance matrix:\n", cov_matrix)
# Interpretation:
# The covariance matrix indicates how the features are correlated.
# A positive covariance suggests that as one feature increases, the other tends to
# increase as well, while a negative covariance suggests an inverse relationship.
# In this case, the covariance matrix shows that the features are positively
# correlated, as both off-diagonal elements are positive.
