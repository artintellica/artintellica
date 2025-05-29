import numpy as np
import torch

# Define two sets of 2D vectors
independent_vectors = np.array([[1, 0], [0, 1]])  # Linearly independent
dependent_vectors = np.array([[1, 2], [2, 4]])  # Linearly dependent


# Convert to PyTorch tensors
ind_vectors_torch = torch.tensor(independent_vectors, dtype=torch.float32)

# Compute Gram matrix
gram_torch = ind_vectors_torch.T @ ind_vectors_torch

# Print result
print("PyTorch Gram matrix (independent):\n", gram_torch.numpy())
