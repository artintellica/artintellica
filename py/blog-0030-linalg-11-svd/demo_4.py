import numpy as np
import torch

# Create a 4x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Rank-1 approximation
k = 1
U_k = U[:, :k]
Sigma_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
A_rank1 = U_k @ Sigma_k @ Vt_k

# Compute Frobenius norm of difference
diff_norm = np.linalg.norm(A - A_rank1, "fro")

# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute SVD
U_torch, S_torch, V_torch = torch.svd(A_torch)

# Print results
print("PyTorch singular values:", S_torch.numpy())
