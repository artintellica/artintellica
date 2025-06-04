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
cluster1 = np.vstack([x1, y1]).T  # Shape: (50, 2)
# Cluster 2: points in another circular pattern, offset
theta2 = np.linspace(0, 2 * np.pi, n_samples // 2)
x2 = 5 + 1 * np.cos(theta2) + np.random.randn(n_samples // 2) * 0.2
y2 = 5 + 1 * np.sin(theta2) + np.random.randn(n_samples // 2) * 0.2
cluster2 = np.vstack([x2, y2]).T  # Shape: (50, 2)
# Combine clusters
X = np.vstack([cluster1, cluster2])  # Shape: (100, 2)
true_labels = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Verify shapes after data generation
print("Data Shape:", X.shape)
print("True Labels Shape:", true_labels.shape)

# Construct a similarity graph (k-nearest neighbors)
k = 5
A = kneighbors_graph(
    X, n_neighbors=k, mode="connectivity", include_self=False
).toarray()
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
spectral_embedding = eigenvectors[:, 1 : k_clusters + 1]
print("Spectral Embedding Shape:", spectral_embedding.shape)

# Apply k-means to the spectral embedding
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(spectral_embedding)
print("Predicted Labels Shape:", predicted_labels.shape)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1], c=predicted_labels, cmap="viridis", label="Predicted Clusters"
)
plt.title("Spectral Clustering Results")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# Compare with true labels
# Since k-means labels may not match true labels directly, try both mappings
accuracy1 = np.mean(predicted_labels == true_labels)
accuracy2 = np.mean(predicted_labels == (1 - true_labels))
accuracy = max(accuracy1, accuracy2)
print("Clustering Accuracy (best matching with true labels):", accuracy)
