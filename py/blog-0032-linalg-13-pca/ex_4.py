import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility (same as Exercise 3)
np.random.seed(44)

# Generate the same synthetic 5D dataset with 100 samples
n_samples = 100
# First dimension (base)
x1 = np.random.randn(n_samples)
# Two dimensions highly correlated with x1
x2 = 0.9 * x1 + np.random.randn(n_samples) * 0.1
x3 = 0.85 * x1 + np.random.randn(n_samples) * 0.15
# Two independent dimensions with some noise
x4 = np.random.randn(n_samples) * 0.5
x5 = np.random.randn(n_samples) * 0.5

# Combine into a 5D dataset
X_5d = np.vstack([x1, x2, x3, x4, x5]).T
print("Shape of synthetic 5D dataset:", X_5d.shape)

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_5d_scaled = scaler.fit_transform(X_5d)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_5d_scaled)
print("Shape of reduced dataset (2D):", X_2d.shape)

# Create synthetic labels (split data into two groups based on the first dimension)
labels = (X_5d[:, 0] > np.median(X_5d[:, 0])).astype(int)
print("Number of points in Group 0:", np.sum(labels == 0))
print("Number of points in Group 1:", np.sum(labels == 1))

# Visualize the 2D projection with different colors for each group
plt.figure(figsize=(8, 6))
plt.scatter(
    X_2d[labels == 0, 0], X_2d[labels == 0, 1], c="blue", alpha=0.6, label="Group 0"
)
plt.scatter(
    X_2d[labels == 1, 0], X_2d[labels == 1, 1], c="orange", alpha=0.6, label="Group 1"
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Projection of 5D Dataset after PCA")
plt.legend()
plt.grid(True)
plt.show()
