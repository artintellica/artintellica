import numpy as np
import torch

# Vectors from Exercise 1 (using same random seed for consistency)
np.random.seed(42)
v1 = np.random.randint(low=-5, high=6, size=2)
v2 = np.random.randint(low=-5, high=6, size=2)
v3 = np.random.randint(low=-5, high=6, size=2)

# Combine vectors into a NumPy matrix (2x3)
vectors_np = np.array([v1, v2, v3]).T  # Transpose to get 2x3 matrix

# Compute Gram matrix using NumPy
gram_np = vectors_np.T @ vectors_np

# Convert vectors to PyTorch tensors
vectors_torch = torch.tensor(vectors_np, dtype=torch.float32)

# Compute Gram matrix using PyTorch
gram_torch = vectors_torch.T @ vectors_torch

# Print results
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Vector v3:", v3)
print("\nVectors as matrix (2x3, NumPy):\n", vectors_np)
print("\nGram matrix (NumPy):\n", gram_np)
print("\nGram matrix (PyTorch):\n", gram_torch.numpy())
print(
    "\nDo NumPy and PyTorch Gram matrices match?",
    np.allclose(gram_np, gram_torch.numpy()),
)
