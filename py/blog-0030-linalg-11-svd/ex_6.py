# Exercise 6: LSA Simulation with SVD

import numpy as np

# Example 4x3 document-term matrix (rows: documents, columns: terms)
A = np.array([[1, 2, 0], [0, 1, 1], [2, 0, 1], [1, 1, 0]])

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Top singular vectors (latent topics)
k = 2  # Number of latent topics to interpret
U_k = U[:, :k]  # Document-topic matrix
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]  # Topic-term matrix

print("Document-term matrix A:\n", A)
print("\nSingular values:", S)
print("\nTop-2 left singular vectors (U_k):\n", U_k)
print("\nTop-2 right singular vectors (Vt_k):\n", Vt_k)
print(
    "\nInterpretation: Rows of U_k show documents' relation to latent topics; columns of Vt_k show terms' relation to topics."
)
