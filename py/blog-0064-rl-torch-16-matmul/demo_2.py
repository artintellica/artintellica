import torch

# A: 2x3 matrix
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# B: 3x2 matrix
B: torch.Tensor = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

# Method 1: Using "@" operator
C1: torch.Tensor = A @ B
print("A @ B:\n", C1)

# Method 2: Using torch.matmul
C2: torch.Tensor = torch.matmul(A, B)
print("torch.matmul(A, B):\n", C2)

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

C3: torch.Tensor = matmul_manual(A, B)
print("Manual matmul(A, B):\n", C3)
print("Equal to PyTorch matmul?", torch.allclose(C1, C3))
