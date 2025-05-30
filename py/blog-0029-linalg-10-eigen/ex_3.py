import numpy as np
import torch

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(42)

# Create the same 3x3 matrix from Exercise 1
A_np = np.random.randint(low=-5, high=6, size=(3, 3))

# Convert to PyTorch tensor
A_torch = torch.tensor(A_np, dtype=torch.float32)

# Compute eigenvalues using NumPy
eigenvalues_np = np.linalg.eigvals(A_np)

# Compute eigenvalues using PyTorch
eigenvalues_torch = torch.linalg.eigvals(A_torch)

# Convert PyTorch eigenvalues to NumPy for comparison
eigenvalues_torch_np = eigenvalues_torch.numpy()

# Verify eigenvalues match
# Sort eigenvalues to handle potential reordering
eigenvalues_np_sorted = np.sort_complex(eigenvalues_np)
eigenvalues_torch_sorted = np.sort_complex(eigenvalues_torch_np)
match = np.allclose(eigenvalues_np_sorted, eigenvalues_torch_sorted, atol=1e-6)

# Print results
print("Matrix A (NumPy):\n", A_np)
print("\nEigenvalues (NumPy):", eigenvalues_np)
print("\nEigenvalues (PyTorch):", eigenvalues_torch_np)
print("\nSorted Eigenvalues (NumPy):", eigenvalues_np_sorted)
print("\nSorted Eigenvalues (PyTorch):", eigenvalues_torch_sorted)
print("\nDo NumPy and PyTorch eigenvalues match?", match)
