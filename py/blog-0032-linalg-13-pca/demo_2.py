import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate a small 2D dataset with some correlation
np.random.seed(42)
n_samples = 100
x1 = np.random.randn(n_samples)
x2 = 0.8 * x1 + np.random.randn(n_samples) * 0.3
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

# Step 4: Project data onto the top principal component (k=1)
k = 1
W = eigenvectors[:, :k]
Z = X_centered @ W
print("Shape of reduced data:", Z.shape)

# # Visualize original and projected data
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")
# # Plot the principal component direction
# scale = 3 * np.sqrt(eigenvalues[0])
# pc1 = mean + scale * eigenvectors[:, 0]
# plt.plot([mean[0], pc1[0]], [mean[1], pc1[1]], "r-", label="PC1 Direction")
# plt.scatter(
#     Z * eigenvectors[0, 0] + mean[0],
#     Z * eigenvectors[1, 0] + mean[1],
#     alpha=0.5,
#     color="green",
#     label="Projected Data (PC1)",
# )
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.title("PCA: Original and Projected Data")
# plt.legend()
# plt.grid(True)
# plt.show()

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=1)
Z_sklearn = pca.fit_transform(X_scaled)
print("Explained variance ratio (PC1):", pca.explained_variance_ratio_)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5, label='Scaled Data')
# Plot the principal component direction from sklearn
mean_scaled = np.mean(X_scaled, axis=0)
scale_sk = 3 * np.sqrt(pca.explained_variance_[0])
pc1_sk = mean_scaled + scale_sk * pca.components_[0]
plt.plot([mean_scaled[0], pc1_sk[0]], [mean_scaled[1], pc1_sk[1]], 'r-', label='PC1 Direction (sklearn)')
plt.xlabel('X1 (scaled)')
plt.ylabel('X2 (scaled)')
plt.title('PCA with scikit-learn')
plt.legend()
plt.grid(True)
plt.show()
