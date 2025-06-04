import numpy as np

# Create a small adjacency matrix for a graph with 5 nodes (a simple connected graph)
# Let's define a graph where node connections form a simple structure (e.g., a path with a cycle)
# A[i,j] = 1 if there is an edge between node i and node j, 0 otherwise
A = np.array(
    [
        [0, 1, 0, 0, 1],  # Node 0 connected to Node 1 and Node 4
        [1, 0, 1, 0, 0],  # Node 1 connected to Node 0 and Node 2
        [0, 1, 0, 1, 0],  # Node 2 connected to Node 1 and Node 3
        [0, 0, 1, 0, 1],  # Node 3 connected to Node 2 and Node 4
        [1, 0, 0, 1, 0],  # Node 4 connected to Node 0 and Node 3
    ]
)
print("Adjacency Matrix (A):")
print(A)

# Compute the degree matrix D (diagonal matrix with node degrees)
# Degree of a node is the sum of its connections (row sum of A)
degrees = np.sum(A, axis=1)
D = np.diag(degrees)
print("\nDegree Matrix (D):")
print(D)

# Compute the Graph Laplacian L = D - A
L = D - A
print("\nGraph Laplacian (L = D - A):")
print(L)

# Verify the relationship L = D - A by checking if D - A equals L
verification = np.allclose(L, D - A)
print("\nVerification: L equals D - A:", verification)
