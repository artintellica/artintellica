import numpy as np

# Use the Laplacian matrix from Exercise 1
# Adjacency matrix for a graph with 5 nodes (a simple connected graph)
A = np.array(
    [
        [0, 1, 0, 0, 1],  # Node 0 connected to Node 1 and Node 4
        [1, 0, 1, 0, 0],  # Node 1 connected to Node 0 and Node 2
        [0, 1, 0, 1, 0],  # Node 2 connected to Node 1 and Node 3
        [0, 0, 1, 0, 1],  # Node 3 connected to Node 2 and Node 4
        [1, 0, 0, 1, 0],  # Node 4 connected to Node 0 and Node 3
    ]
)

# Compute the degree matrix D and Laplacian L = D - A
degrees = np.sum(A, axis=1)
D = np.diag(degrees)
L = D - A

print("Graph Laplacian (L):")
print(L)

# Compute eigenvalues and eigenvectors using np.linalg.eigh
# np.linalg.eigh is used for symmetric matrices, ensuring real eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Print eigenvalues
print("\nEigenvalues of Laplacian:")
print(eigenvalues)

# Print eigenvectors (each column is an eigenvector)
print("\nEigenvectors of Laplacian (each column corresponds to an eigenvalue):")
print(eigenvectors)

# Check if the smallest eigenvalue is close to 0 (indicating a connected graph)
smallest_eigenvalue = eigenvalues[0]
is_connected = np.isclose(smallest_eigenvalue, 0, atol=1e-8)
print("\nSmallest Eigenvalue:", smallest_eigenvalue)
print("Is the graph connected (smallest eigenvalue close to 0)?", is_connected)
