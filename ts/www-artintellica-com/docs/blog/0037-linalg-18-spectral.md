+++
title = "Linear Algebra for Machine Learning, Part 18: Spectral Methods in ML (Graph Laplacians, etc.)"
author = "Artintellica"
date = "2025-07-15"
+++

Welcome back to our series on linear algebra for machine learning! In this post, we’re exploring **Spectral Methods in ML**, focusing on tools like Graph Laplacians that leverage the eigenvalues and eigenvectors of matrices to solve problems in clustering, graph-based machine learning, and signal processing. Spectral methods provide a powerful way to uncover structure in data by transforming it into a spectral domain. Whether you're working on community detection in networks or dimensionality reduction, understanding these techniques is invaluable. Let’s dive into the math, intuition, and implementation with Python code using NumPy and SciPy, visualizations, and hands-on exercises.

## What Are Spectral Methods in ML?

Spectral methods in machine learning use the eigenvalues and eigenvectors (the "spectrum") of matrices associated with data to perform tasks like clustering, classification, and embedding. These methods often involve constructing a matrix that captures relationships in the data—such as similarity or adjacency—and then analyzing its spectral properties.

### Graph Laplacians

A key concept in spectral methods is the **Graph Laplacian**, derived from a graph’s adjacency matrix. Given a graph with \( n \) nodes, we define:
- **Adjacency Matrix** \( A \in \mathbb{R}^{n \times n} \): \( A_{ij} = w_{ij} \) if there is an edge between nodes \( i \) and \( j \) with weight \( w_{ij} \), otherwise 0.
- **Degree Matrix** \( D \in \mathbb{R}^{n \times n} \): A diagonal matrix where \( D_{ii} = \sum_j A_{ij} \), the degree of node \( i \).
- **Graph Laplacian** \( L \in \mathbb{R}^{n \times n} \): Defined as \( L = D - A \). It is symmetric and positive semi-definite, ensuring real, non-negative eigenvalues.

The Laplacian’s eigenvalues and eigenvectors reveal structural properties of the graph. For instance, the smallest eigenvalue is always 0 (with an eigenvector of all 1s for connected graphs), and the number of zero eigenvalues corresponds to the number of connected components. The eigenvectors associated with the smallest non-zero eigenvalues (the "Fiedler vectors") provide information about how to partition the graph.

### Spectral Clustering

One prominent application is **spectral clustering**, which uses the spectrum of the Laplacian to cluster data points. The process involves:
1. Constructing a similarity graph (e.g., k-nearest neighbors) and its Laplacian \( L \).
2. Computing the first \( k \) eigenvectors corresponding to the smallest eigenvalues of \( L \).
3. Using these eigenvectors as a low-dimensional representation of the data and applying a clustering algorithm like k-means.

Mathematically, spectral clustering minimizes the "cut" between clusters while balancing cluster sizes, often outperforming traditional clustering on non-linearly separable data.

## Why Do Spectral Methods Matter in Machine Learning?

Spectral methods are vital in machine learning for several reasons:
1. **Graph-Based Learning**: They enable analysis of data with relational structure, such as social networks, citation networks, and molecular graphs.
2. **Clustering**: Spectral clustering can capture complex, non-linear structures in data, unlike methods like k-means that assume spherical clusters.
3. **Signal Processing**: Spectral methods decompose signals on graphs, useful in domains like image processing and time series analysis.
4. **Dimensionality Reduction**: They provide embeddings (e.g., via Laplacian eigenmaps) that preserve structural relationships for visualization or downstream tasks.

Understanding the linear algebra of spectral methods equips you to tackle problems where data relationships are as critical as the data itself.

## Implementing Spectral Methods with Python

Let’s implement spectral clustering using the Graph Laplacian with NumPy and SciPy. We’ll also visualize the results to build intuition about how eigenvectors reveal data structure.

### Example 1: Spectral Clustering with Graph Laplacian

```python
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D data with two clusters (non-linearly separable)
n_samples = 100
# Cluster 1: points in a circular pattern
theta1 = np.linspace(0, 2 * np.pi, n_samples // 2)
x1 = 2 * np.cos(theta1) + np.random.randn(n_samples // 2) * 0.2
y1 = 2 * np.sin(theta1) + np.random.randn(n_samples // 2) * 0.2
# Cluster 2: points in another circular pattern, offset
theta2 = np.linspace(0, 2 * np.pi, n_samples // 2)
x2 = 5 + 1 * np.cos(theta2) + np.random.randn(n_samples // 2) * 0.2
y2 = 5 + 1 * np.sin(theta2) + np.random.randn(n_samples // 2) * 0.2
X = np.vstack([(x1, y1), (x2, y2)]).T
true_labels = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Construct a similarity graph (k-nearest neighbors)
k = 5
A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False).toarray()
A = 0.5 * (A + A.T)  # Ensure symmetry
print("Adjacency Matrix Shape:", A.shape)

# Compute degree matrix D and Laplacian L = D - A
D = np.diag(A.sum(axis=1))
L = D - A
print("Laplacian Matrix Shape:", L.shape)

# Compute the first k eigenvectors of L (smallest eigenvalues)
k_clusters = 2
eigenvalues, eigenvectors = np.linalg.eigh(L)
# Select the first k eigenvectors (excluding the first one if graph is connected)
spectral_embedding = eigenvectors[:, 1:k_clusters + 1]
print("Spectral Embedding Shape:", spectral_embedding.shape)

# Apply k-means to the spectral embedding
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(spectral_embedding)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', label='Predicted Clusters')
plt.title('Spectral Clustering Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Compare with true labels
accuracy = np.mean(predicted_labels == true_labels)
print("Clustering Accuracy (matching true labels):", accuracy)
```

**Output (abbreviated)**:
```
Adjacency Matrix Shape: (100, 100)
Laplacian Matrix Shape: (100, 100)
Spectral Embedding Shape: (100, 2)
Clustering Accuracy (matching true labels): 0.98
```

This example generates a synthetic 2D dataset with two non-linearly separable clusters (circular patterns) and applies spectral clustering. It constructs a k-nearest neighbors graph as the adjacency matrix, computes the Graph Laplacian, and finds its eigenvectors corresponding to the smallest eigenvalues. These eigenvectors form a low-dimensional embedding, which is clustered using k-means. The visualization shows the predicted clusters, and the accuracy (compared to true labels) demonstrates the effectiveness of spectral clustering in capturing non-linear structure.

## Exercises

Here are six exercises to deepen your understanding of spectral methods in machine learning. Each exercise requires writing Python code to explore concepts and applications using NumPy and SciPy.

1. **Graph Laplacian Construction**: Create a small adjacency matrix for a graph with 5 nodes (e.g., a simple connected graph). Compute the degree matrix and Graph Laplacian manually using NumPy, and print all three matrices to verify the relationship \( L = D - A \).
2. **Eigenvalues of Laplacian**: Using the Laplacian from Exercise 1, compute its eigenvalues and eigenvectors with `np.linalg.eigh`. Print the eigenvalues and check if the smallest eigenvalue is close to 0 (indicating a connected graph).
3. **Spectral Embedding Visualization**: Generate a synthetic 2D dataset with 3 clusters (e.g., using Gaussian distributions). Compute the Graph Laplacian (k=5 neighbors) and its first 2 eigenvectors for spectral embedding. Plot the embedding to visualize the separation of clusters.
4. **Spectral Clustering on Synthetic Data**: Extend Exercise 3 by applying k-means to the spectral embedding with 3 clusters. Plot the original data points colored by predicted clusters and compare with true labels.
5. **Graph Connectivity Analysis**: Create an adjacency matrix for a graph with 2 disconnected components (e.g., two separate clusters of nodes). Compute the Laplacian and its eigenvalues. Print the number of eigenvalues close to 0 to infer the number of connected components.
6. **Real Data Spectral Clustering**: Load the Iris dataset from scikit-learn (`sklearn.datasets.load_iris`), compute a k-nearest neighbors graph (k=5), and apply spectral clustering with 3 clusters. Compute the clustering accuracy against true labels and visualize the results in 2D using the first two features.

## Conclusion

Spectral Methods in ML, powered by tools like Graph Laplacians, offer a profound way to uncover hidden structure in data through the lens of linear algebra. By leveraging eigenvalues and eigenvectors, these methods excel in clustering, graph analysis, and signal processing, often capturing relationships that traditional approaches miss. Our implementation of spectral clustering with NumPy and SciPy demonstrates the practical impact of these concepts.

In the next post, we’ll explore **Kernel Methods and Feature Spaces**, diving into how linear algebra enables non-linear learning through the kernel trick and feature transformations. Stay tuned, and happy learning!
