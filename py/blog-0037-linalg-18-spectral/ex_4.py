import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D dataset with 3 clusters using Gaussian distributions (same as Exercise 3)
n_samples_per_cluster = 50
total_samples = n_samples_per_cluster * 3

# Cluster 1: Centered at (-2, -2)
cluster1 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([-2, -2])
# Cluster 2: Centered at (2, -2)
cluster2 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([2, -2])
# Cluster 3: Centered at (0, 2)
cluster3 = np.random.randn(n_samples_per_cluster, 2) * 0.5 + np.array([0, 2])

# Combine into a single dataset
X = np.vstack([cluster1, cluster2, cluster3])
true_labels = np.hstack(
    [
        np.zeros(n_samples_per_cluster),
        np.ones(n_samples_per_cluster),
        np.full(n_samples_per_cluster, 2),
    ]
)

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
k_clusters = 3
eigenvalues, eigenvectors = np.linalg.eigh(L)
# Select the first k eigenvectors after the smallest one (excluding the trivial eigenvector)
spectral_embedding = eigenvectors[
    :, 1 : k_clusters + 1
]  # First k non-trivial eigenvectors
print("Spectral Embedding Shape:", spectral_embedding.shape)

# Apply k-means to the spectral embedding with 3 clusters
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(spectral_embedding)
print("Predicted Labels Shape:", predicted_labels.shape)

# Plot the original data with true and predicted labels
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap="viridis", label="True Clusters")
plt.title("Original Data with True Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Cluster")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(
    X[:, 0], X[:, 1], c=predicted_labels, cmap="viridis", label="Predicted Clusters"
)
plt.title("Original Data with Predicted Clusters (Spectral Clustering)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Cluster")
plt.grid(True)

plt.tight_layout()
plt.show()


# Compute clustering accuracy (accounting for label permutation)
# Since k-means labels are arbitrary, we try all possible mappings to true labels
def compute_clustering_accuracy(true_labels, pred_labels, k_clusters):
    accuracies = []
    from itertools import permutations

    for perm in permutations(range(k_clusters)):
        mapped_labels = np.array([perm[l] for l in pred_labels])
        acc = np.mean(mapped_labels == true_labels)
        accuracies.append(acc)
    return max(accuracies)


accuracy = compute_clustering_accuracy(true_labels, predicted_labels, k_clusters)
print("Clustering Accuracy (best matching with true labels):", accuracy)
