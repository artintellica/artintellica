import numpy as np
from sklearn.metrics import pairwise_distances

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic high-dimensional data (200 samples, 500 dimensions)
n_samples, d = 200, 500
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Target reduced dimension
k = 50  # Reducing to 50 dimensions as specified
print("Target Reduced Dimension (k):", k)

# Create random projection matrix (Gaussian entries)
R = np.random.randn(d, k) / np.sqrt(k)  # Scale to preserve distances approximately
print("Random Projection Matrix Shape:", R.shape)

# Project data to lower dimension
X_proj = X @ R
print("Projected Data Shape:", X_proj.shape)

# Compute pairwise distances before and after projection
dist_original = pairwise_distances(X, metric="euclidean")
dist_projected = pairwise_distances(X_proj, metric="euclidean")

# Compute relative distortion, avoiding division by zero
small_value = 1e-10  # Small constant to prevent division by zero
relative_distortion = np.abs(dist_projected - dist_original) / (
    dist_original + small_value
)
mean_distortion = np.mean(relative_distortion[np.isfinite(relative_distortion)])
print("Mean Relative Distortion:", mean_distortion)
