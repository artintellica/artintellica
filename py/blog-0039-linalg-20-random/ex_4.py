import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics import pairwise_distances

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (128 samples, 128 features)
n_samples, d = 128, 128  # Hadamard matrix requires dimensions as powers of 2
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Create a Hadamard matrix (structured orthogonal matrix)
H = hadamard(d) / np.sqrt(d)  # Normalize to preserve distances approximately
print("Hadamard Matrix Shape:", H.shape)

# Randomly select a subset of columns for projection (reduce to k=32 dimensions)
k = 32
indices = np.random.choice(d, k, replace=False)
R_structured = H[:, indices]
print("Structured Projection Matrix Shape:", R_structured.shape)

# Project data to lower dimension
X_proj = X @ R_structured
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
