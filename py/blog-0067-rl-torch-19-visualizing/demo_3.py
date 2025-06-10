import torch
import matplotlib.pyplot as plt
import math

# Generate 2D Gaussian blobs
torch.manual_seed(42)
N = 200
mean = torch.tensor([2.0, -3.0])
cov = torch.tensor([[3.0, 1.2], [1.2, 2.0]])
L = torch.linalg.cholesky(cov)
data = torch.randn(N, 2) @ L.T + mean


def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float32,
    )


def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)


# Transform: rotate 45°, scale x=0.5, y=2
theta = math.radians(45)
R = rotation_matrix(theta)
S = scaling_matrix(0.5, 2.0)
transformed_data = (data @ S.T) @ R.T  # scale, then rotate

# Mean vector
mean_vec: torch.Tensor = data.mean(dim=0)
print("Mean vector:\n", mean_vec)

# Centered data
data_centered: torch.Tensor = data - mean_vec

# Covariance matrix
cov_matrix: torch.Tensor = (data_centered.T @ data_centered) / data.shape[0]
print("Covariance matrix:\n", cov_matrix)
