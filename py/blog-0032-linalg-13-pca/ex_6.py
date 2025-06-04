import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # 4D data (sepal length, sepal width, petal length, petal width)
print("Shape of original Iris dataset:", X.shape)

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Shape of scaled dataset:", X_scaled.shape)

# Apply PCA to reduce from 4D to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print("Shape of reduced dataset (2D):", X_2d.shape)

# Reconstruct the data from the 2D representation
# X_reconstructed = X_2d @ pca.components_ + pca.mean_ (but since we scaled, use inverse transform)
X_reconstructed_scaled = pca.inverse_transform(X_2d)
print("Shape of reconstructed scaled dataset:", X_reconstructed_scaled.shape)

# Inverse transform the scaling to get back to original space
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)
print("Shape of reconstructed original dataset:", X_reconstructed.shape)

# Compute the mean squared error (MSE) between original and reconstructed data
mse = np.mean((X - X_reconstructed) ** 2)
print("\nMean Squared Error between original and reconstructed data:", mse)

# Compute MSE per feature for detailed insight
mse_per_feature = np.mean((X - X_reconstructed) ** 2, axis=0)
print("\nMean Squared Error per feature:")
feature_names = iris.feature_names
for i, error in enumerate(mse_per_feature):
    print(f"{feature_names[i]}: {error:.6f}")

# Print explained variance ratio to relate MSE to information loss
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance ratio for top 2 components:")
print(f"Principal Component 1: {explained_variance_ratio[0]:.4f}")
print(f"Principal Component 2: {explained_variance_ratio[1]:.4f}")
print(f"Total explained variance (PC1 + PC2): {sum(explained_variance_ratio):.4f}")
