import numpy as np

# Create an adjacency matrix for a graph with 2 disconnected components
# We'll define a graph with 6 nodes: two separate components (3 nodes each)
# Component 1: Nodes 0, 1, 2 connected as a triangle
# Component 2: Nodes 3, 4, 5 connected as a triangle
A = np.array([
    [0, 1, 1, 0, 0, 0],  # Node 0 connected to 1 and 2
    [1, 0, 1, 0, 0, 0],  # Node 1 connected to 0 and 2
    [1, 1, 0, 0, 0, 0],  # Node 2 connected to 0 and 1
    [0, 0, 0, 0, 1, 1],  # Node 3 connected to 4 and 5
    [0, 0, 0, 1, 0, 1],  # Node 4 connected to 3 and 5
    [0, 0, 0, 1, 1, 0]   # Node 5 connected to 3 and 4
])
print("Adjacency Matrix (A):")
print(A)

# Compute the degree matrix D (diagonal matrix with node degrees)
degrees = np.sum(A, axis=1)
D = np.diag(degrees)
print("\nDegree Matrix (D):")
print(D)

# Compute the Graph Laplacian L = D - A
L = D - A
print("\nGraph Laplacian (L = D - A):")
print(L)

# Compute eigenvalues and eigenvectors of the Laplacian using np.linalg.eigh
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Print eigenvalues
print("\nEigenvalues of Laplacian:")
print(eigenvalues)

# Count the number of eigenvalues close to 0 to infer the number of connected components
tolerance = 1e-8  # Small tolerance for numerical precision
num_connected_components = np.sum(np.abs(eigenvalues) < tolerance)
print("\nNumber of eigenvalues close to 0 (indicating connected components):", num_connected_components)
print("Conclusion: The graph has", num_connected_components, "connected components.")
