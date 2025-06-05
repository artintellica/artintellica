import numpy as np
from scipy.linalg import hadamard

# Set random seed for reproducibility
np.random.seed(42)

# Generate smaller synthetic data for demonstration (64 samples, 64 dimensions)
n_samples, d = 64, 64  # Hadamard matrix requires power of 2
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Create a Hadamard matrix (structured orthogonal matrix)
H = hadamard(d) / np.sqrt(d)  # Normalize to preserve distances approximately
print("Hadamard Matrix Shape:", H.shape)

# Randomly select a subset of columns for projection (reduce to k=16 dimensions)
k = 16
indices = np.random.choice(d, k, replace=False)
R_structured = H[:, indices]
print("Structured Projection Matrix Shape:", R_structured.shape)

# Project data to lower dimension
X_proj_structured = X @ R_structured
print("Projected Data Shape:", X_proj_structured.shape)
