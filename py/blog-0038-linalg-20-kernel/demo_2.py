import numpy as np
from sklearn.datasets import make_moons

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D data with two non-linearly separable classes (moon shapes)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
print("Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Compute a custom RBF kernel (Gram) matrix manually
def rbf_kernel(X1, X2, sigma=1.0):
    # Compute squared Euclidean distances between all pairs of points
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    sq_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    # Apply RBF kernel formula
    return np.exp(-sq_dist / (2 * sigma ** 2))

# Use a subset of the data for demonstration
X_subset = X[:10]
K = rbf_kernel(X_subset, X_subset, sigma=1.0)
print("\nRBF Kernel (Gram) Matrix for Subset (10x10):")
print("Shape:", K.shape)
print(K)
