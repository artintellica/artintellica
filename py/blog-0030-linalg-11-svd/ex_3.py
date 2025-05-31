# Exercise 3: PyTorch SVD and singular value comparison

import numpy as np
import torch

# Use the matrix from Exercise 1 (replace this with your actual matrix if needed)
np.random.seed(0)
A = np.random.randint(-5, 6, size=(3, 4))

# NumPy SVD
_, S_np, _ = np.linalg.svd(A, full_matrices=False)

# PyTorch SVD
A_torch = torch.tensor(A, dtype=torch.float32)
U_torch, S_torch, V_torch = torch.svd(A_torch)

print("NumPy singular values:", S_np)
print("PyTorch singular values:", S_torch.numpy())
print("Singular values match?", np.allclose(S_np, S_torch.numpy(), atol=1e-5))
