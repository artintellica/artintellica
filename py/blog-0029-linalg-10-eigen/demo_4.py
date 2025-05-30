import numpy as np
import torch

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 2]])


# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute eigenvalues
eigenvalues_torch = torch.linalg.eigvals(A_torch)

# Print results
print("PyTorch eigenvalues:", eigenvalues_torch.numpy())
