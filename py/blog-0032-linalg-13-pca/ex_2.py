import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(43)

# Generate the same 2D dataset with 50 points and some correlation
n_samples = 50
x1 = np.random.randn(n_samples)
x2 = 0.7 * x1 + np.random.randn(n_samples) * 0.4
X = np.vstack([x1, x2]).T
print("Original dataset shape:", X.shape)

# Step 1: Center the data (for manual PCA)
mean = np.mean(X, axis=0)
X_centered = X - mean

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

# Step 4: Compute explained variance ratio manually
total_variance = np.sum(eigenvalues)
explained_variance_ratio_manual = eigenvalues / total_variance
print("\nManually computed explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio_manual):
    print(f"Principal Component {i+1}: {ratio:.4f}")

# Step 5: Compare with scikit-learn's PCA
# Standardize the data (mean=0, variance=1) as sklearn PCA assumes standardized input for comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with scikit-learn
pca = PCA(n_components=2)
pca.fit(X_scaled)
explained_variance_ratio_sklearn = pca.explained_variance_ratio_
print("\nscikit-learn explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio_sklearn):
    print(f"Principal Component {i+1}: {ratio:.4f}")

# Compare the results
print("\nComparison of explained variance ratios:")
for i in range(len(explained_variance_ratio_manual)):
    print(
        f"PC{i+1} - Manual: {explained_variance_ratio_manual[i]:.4f}, sklearn: {explained_variance_ratio_sklearn[i]:.4f}"
    )
