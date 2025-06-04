import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate a 5D dataset
np.random.seed(42)
n_samples = 100
X_5d = np.random.randn(n_samples, 5)
X_5d[:, 1] = 0.8 * X_5d[:, 0] + 0.2 * np.random.randn(n_samples)
X_5d[:, 2] = 0.5 * X_5d[:, 0] + 0.3 * np.random.randn(n_samples)

# Apply PCA
scaler = StandardScaler()
X_5d_scaled = scaler.fit_transform(X_5d)
pca_5d = PCA()
pca_5d.fit(X_5d_scaled)

# Plot cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), np.cumsum(pca_5d.explained_variance_ratio_), marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Cumulative Explained Variance by Principal Components")
plt.grid(True)
plt.show()
