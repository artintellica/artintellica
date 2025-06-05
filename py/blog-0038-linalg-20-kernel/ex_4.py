import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load the Iris dataset
iris = load_iris()
X = iris.data  # 4D features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Class labels (0: Setosa, 1: Versicolor, 2: Virginica)

print("Original Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Apply standard PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply Kernel PCA with RBF kernel to reduce to 2D
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, random_state=42)
X_kpca = kpca.fit_transform(X)


# Function to plot 2D data with labels
def plot_2d_data(X_2d, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y,
        palette="deep",
        style=y,
        markers=["o", "s", "^"],
        s=100,
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.legend(title="Class", labels=iris.target_names)
    plt.show()


# Plot results for standard PCA
plot_2d_data(X_pca, y, "Standard PCA on Iris Dataset (2D)")

# Plot results for Kernel PCA with RBF kernel
plot_2d_data(X_kpca, y, "Kernel PCA (RBF Kernel) on Iris Dataset (2D)")

# Print explained variance ratio for standard PCA (for reference)
explained_variance_ratio_pca = pca.explained_variance_ratio_
print("Explained Variance Ratio (Standard PCA):", explained_variance_ratio_pca)
print("Total Explained Variance (Standard PCA):", sum(explained_variance_ratio_pca))
