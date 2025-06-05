import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic high-dimensional data (100 samples, 1000 dimensions)
n_samples, d = 100, 1000
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Target reduced dimension (based on Johnson-Lindenstrauss, k ~ log(n)/epsilon^2)
epsilon = 0.1  # Distortion factor
k_calculated = int(8 * np.log(n_samples) / (epsilon**2))  # Rough estimate
# Cap k to be smaller than original dimension d for practical reduction
k = min(k_calculated, 100)  # Reducing to 100 dimensions for demo purposes
print("Calculated Target Reduced Dimension:", k_calculated)
print("Adjusted Target Reduced Dimension (k):", k)

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
# Add small epsilon to denominator to prevent division by zero
small_value = 1e-10
relative_distortion = np.abs(dist_projected - dist_original) / (
    dist_original + small_value
)
mean_distortion = np.mean(relative_distortion[np.isfinite(relative_distortion)])
print("Mean Relative Distortion:", mean_distortion)

# Visualize distortion distribution
plt.figure(figsize=(8, 6))
plt.hist(
    relative_distortion.flatten(),
    bins=50,
    density=True,
    alpha=0.7,
    color="blue",
    range=(0, 1),
)
plt.title("Distribution of Relative Distortion in Pairwise Distances")
plt.xlabel("Relative Distortion")
plt.ylabel("Density")
plt.grid(True)
plt.show()
