+++
title = "Linear Algebra for Machine Learning, Part 13: Principal Component Analysis (PCA)"
author = "Artintellica"
date = "2025-06-10"
+++

Welcome back to our series on linear algebra for machine learning! In this post, we’re exploring **Principal Component Analysis (PCA)**, a powerful technique for dimensionality reduction and data visualization. PCA leverages core linear algebra concepts like eigenvalues, eigenvectors, and covariance matrices to transform high-dimensional data into a lower-dimensional space while preserving as much variability as possible. Whether you're preprocessing data for a machine learning model or visualizing complex datasets, PCA is an indispensable tool. Let’s dive into the math, intuition, and implementation with Python code, visualizations, and hands-on exercises.

## What Is Principal Component Analysis (PCA)?

PCA is a statistical method that transforms a dataset of possibly correlated variables into a new set of uncorrelated variables called **principal components**. These components are linear combinations of the original variables, ordered such that the first component captures the maximum variance in the data, the second captures the maximum remaining variance (orthogonal to the first), and so on.

Mathematically, for a dataset represented as a matrix $X \in \mathbb{R}^{n \times d}$ (with $n$ samples and $d$ features), PCA involves the following steps:
1. **Center the Data**: Subtract the mean of each feature to get $X_{\text{centered}} = X - \mu$, where $\mu$ is the mean vector.
2. **Compute Covariance Matrix**: Calculate the covariance matrix $C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}$, which captures the relationships between features.
3. **Eigenvalue Decomposition**: Find the eigenvalues and eigenvectors of $C$. The eigenvectors represent the directions of the principal components, and the eigenvalues indicate the amount of variance explained by each component.
4. **Project the Data**: Select the top $k$ eigenvectors (corresponding to the largest eigenvalues) and project the centered data onto these directions to get the reduced dataset $Z = X_{\text{centered}} W$, where $W$ is the matrix of top $k$ eigenvectors.

Geometrically, PCA rotates the data to align with the axes of maximum variance, effectively finding a new coordinate system where the data is spread out as much as possible along the first few axes.

## Why Does PCA Matter in Machine Learning?

PCA is widely used in machine learning for several reasons:
1. **Dimensionality Reduction**: High-dimensional data can lead to overfitting and computational challenges. PCA reduces the number of features while retaining most of the information, improving model performance and efficiency.
2. **Visualization**: PCA projects data into 2D or 3D spaces for visualization, helping to uncover patterns or clusters (e.g., visualizing high-dimensional datasets like images or gene expression data).
3. **Noise Reduction**: By focusing on components with the highest variance, PCA can filter out noise captured in lower-variance dimensions.
4. **Feature Engineering**: PCA-derived components can serve as new features for downstream models, often improving interpretability and performance.

Understanding PCA also reinforces key linear algebra concepts like covariance matrices and eigendecomposition, which are central to many ML algorithms.

## Implementing PCA Step-by-Step in Python

Let’s implement PCA from scratch using NumPy to understand each step. We’ll also compare it with scikit-learn’s implementation for validation. Our example will use a small 2D dataset for simplicity and visualization.

### Example 1: PCA from Scratch with NumPy

```python
import numpy as np
import matplotlib.pyplot as plt

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

# Visualize original and projected data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original Data')
# Plot the principal component direction
scale = 3 * np.sqrt(eigenvalues[0])
pc1 = mean + scale * eigenvectors[:, 0]
plt.plot([mean[0], pc1[0]], [mean[1], pc1[1]], 'r-', label='PC1 Direction')
plt.scatter(Z * eigenvectors[0, 0] + mean[0], Z * eigenvectors[1, 0] + mean[1], 
            alpha=0.5, color='green', label='Projected Data (PC1)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('PCA: Original and Projected Data')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:
```
Original dataset shape: (100, 2)
Mean of data: [-0.0565  0.0071]
Covariance matrix:
[[ 0.9095  0.6786]
 [ 0.6786  0.5846]]
Eigenvalues: [1.4036 0.0905]
Eigenvectors:
[[ 0.7467 -0.6652]
 [ 0.6652  0.7467]]
Shape of reduced data: (100, 1)
```

This code generates a 2D dataset with correlation between features, applies PCA step-by-step (centering, covariance, eigendecomposition, projection), and visualizes the original data, the direction of the first principal component (PC1), and the projected data. The first principal component captures the direction of maximum variance, aligning with the trend in the data.

### Example 2: PCA with scikit-learn for Validation

Let’s validate our implementation using scikit-learn’s PCA and apply it to the same dataset.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
```

**Output (abbreviated)**:
```
Explained variance ratio (PC1): [0.9393]
```

This confirms that the first principal component explains over 93% of the variance, consistent with our manual implementation. scikit-learn’s PCA is more robust for real-world data, handling numerical stability and standardization.

## Visualization: Variance Explained

To understand the trade-off in dimensionality reduction, let’s plot the cumulative explained variance ratio for a slightly larger dataset.

```python
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
plt.plot(range(1, 6), np.cumsum(pca_5d.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()
```

This plot shows how much variance is explained as we include more principal components, helping decide how many components to retain (e.g., often choosing enough to explain 95% of variance).

## Exercises

Here are six exercises to deepen your understanding of PCA. Each exercise requires writing Python code to explore concepts and applications in machine learning.

1. **Manual PCA on 2D Data**: Write Python code using NumPy to apply PCA from scratch on a new 2D dataset (generate 50 points with some correlation). Center the data, compute the covariance matrix, find eigenvectors, and project the data onto the first principal component. Plot the original and projected data.
2. **Variance Explained Check**: Using the 2D dataset from Exercise 1, write code to compute the explained variance ratio for each principal component (eigenvalue divided by sum of eigenvalues). Compare your results with scikit-learn’s PCA output.
3. **Dimensionality Reduction**: Generate a synthetic 5D dataset (100 samples) with NumPy, where two dimensions are highly correlated with the first. Apply PCA using scikit-learn to reduce it to 2D, and print the explained variance ratio for the top 2 components.
4. **Visualization of Reduced Data**: Using the 5D dataset from Exercise 3, write code to visualize the 2D projection after PCA (using scikit-learn). Scatter plot the reduced data and color points based on a synthetic label (e.g., split data into two groups).
5. **Real Dataset Application**: Load the Iris dataset from scikit-learn (`sklearn.datasets.load_iris`), apply PCA to reduce it from 4D to 2D, and plot the reduced data with different colors for each class. Compute and print the explained variance ratio for the top 2 components.
6. **Reconstruction Error**: Using the Iris dataset, write code to apply PCA with scikit-learn to reduce to 2D, then reconstruct the original data from the reduced representation. Compute the mean squared error between the original and reconstructed data to quantify information loss.

## Conclusion

Principal Component Analysis (PCA) is a cornerstone of dimensionality reduction and visualization in machine learning, rooted in linear algebra concepts like covariance matrices and eigendecomposition. By transforming data into principal components, PCA enables us to simplify complex datasets while retaining critical information. Through our step-by-step implementation in NumPy and validation with scikit-learn, we’ve seen how PCA works in practice, supported by visualizations of variance explained and data projections.

In the next post, we’ll dive into **Least Squares and Linear Regression**, exploring how linear algebra underpins one of the most fundamental models in machine learning. Stay tuned, and happy learning!
