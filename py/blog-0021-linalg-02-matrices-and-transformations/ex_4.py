import numpy as np
import torch

# Define the scaling matrix and vector (from Exercise 3)
S_np = np.array([[2, 0], [0, 3]])
vector_np = np.array([1, 1])

# Convert to PyTorch tensors
S_torch = torch.tensor(S_np, dtype=torch.float32)
vector_torch = torch.tensor(vector_np, dtype=torch.float32)

# Perform matrix-vector multiplication using PyTorch
scaled_vector_torch = S_torch @ vector_torch

# Perform matrix-vector multiplication using NumPy for verification
scaled_vector_np = S_np @ vector_np

# Print results
print("Scaling matrix (NumPy):\n", S_np)
print("Original vector (NumPy):", vector_np)
print("Scaled vector (PyTorch):", scaled_vector_torch.numpy())
print("Scaled vector (NumPy):", scaled_vector_np)

# Verify the results match
match = np.allclose(scaled_vector_torch.numpy(), scaled_vector_np)
print("Do PyTorch and NumPy results match?", match)
