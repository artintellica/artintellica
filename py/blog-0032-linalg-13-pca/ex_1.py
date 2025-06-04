import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(43)

# Generate a new 2D dataset with 50 points and some correlation
n_samples = 50
x1 = np.random.randn(n_samples)
x2 = 0.7 * x1 + np.random.randn(n_samples) * 0.4
X = np.vstack([x1, x2]).T
print("Original dataset shape:", X.shape)

# Step 1: Center the data
mean = np.mean(X, axis=0)
X_centered = X - mean
print("Mean of data:", mean)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T, bias=False)
print("Covariance matrix:")
print(cov_matrix)

# Step 3: Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Step 4: Project data onto the first principal component (k=1)
k = 1
W = eigenvectors[:, :k]
Z = X_centered @ W
print("Shape of reduced data:", Z.shape)

# Visualize original and projected data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")
# Plot the principal component direction
scale = 3 * np.sqrt(eigenvalues[0])
pc1 = mean + scale * eigenvectors[:, 0]
plt.plot([mean[0], pc1[0]], [mean[1], pc1[1]], "r-", label="PC1 Direction")
# Plot projected data (reconstruct points along PC1)
projected_points = Z @ W.T + mean
plt.scatter(
    projected_points[:, 0],
    projected_points[:, 1],
    alpha=0.5,
    color="green",
    label="Projected Data (PC1)",
)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Manual PCA: Original and Projected Data")
plt.legend()
plt.grid(True)
plt.show()
