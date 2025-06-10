import torch

# - Implement matrix multiplication manually using nested loops.
# - Compare the manual result with PyTorch’s builtin `@`; confirm they are
#   identical.
def matmul_manual(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Matrix dimensions do not match!"
    C = torch.zeros((m, p), dtype=A.dtype)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Example matrices
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B: torch.Tensor = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
# Manual matrix multiplication
C_manual: torch.Tensor = matmul_manual(A, B)
# PyTorch matrix multiplication using "@"
C_builtin: torch.Tensor = A @ B
# Check if the results are equal
are_equal: bool = torch.allclose(C_manual, C_builtin)
# Print results
print("Manual multiplication result:\n", C_manual)
print("PyTorch multiplication result (using @):\n", C_builtin)
print("Are the results equal?", are_equal)

