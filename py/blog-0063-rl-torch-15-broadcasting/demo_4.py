import torch

A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
row: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])  # Shape: (3,)

# Manual using loops (for small examples)
manual_sum: torch.Tensor = torch.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        manual_sum[i, j] = A[i, j] + row[j]

print("Manual elementwise sum:\n", manual_sum)

# PyTorch broadcasting (just one line)
broadcast_sum: torch.Tensor = A + row
print("Broadcast sum:\n", broadcast_sum)

# Confirm equality
print("Are they equal?", torch.allclose(manual_sum, broadcast_sum))

