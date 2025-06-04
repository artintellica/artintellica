import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(44)

# Generate a synthetic 5D dataset with 100 samples
n_samples = 100
# First dimension (base)
dim1 = np.random.randn(n_samples)
# Two dimensions highly correlated with dim1
dim2 = 0.9 * dim1 + np.random.randn(n_samples) * 0.1
dim3 = 0.85 * dim1 + np.random.randn(n_samples) * 0.15
# Two other dimensions with less correlation or independent
dim4 = 0.3 * dim1 + np.random.randn(n_samples) * 0.7
dim5 = np.random.randn(n_samples)

# Combine into a 5D dataset
X_5d = np.vstack([dim1, dim2, dim3, dim4, dim5]).T
print("Shape of synthetic 5D dataset:", X_5d.shape)

# Standardize the data (mean=0, variance=1) as recommended for PCA
scaler = StandardScaler()
X_5d_scaled = scaler.fit_transform(X_5d)
print("Shape of standardized dataset:", X_5d_scaled.shape)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_5d_scaled)
print("Shape of reduced 2D dataset:", X_2d.shape)

# Print explained variance ratio for the top 2 components
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance ratio for top 2 components:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i+1}: {ratio:.4f}")
print("Total explained variance by top 2 components:", sum(explained_variance_ratio))
