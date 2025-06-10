import torch

# - For a $2 \times 3$ matrix `P` and a row vector `w` of length 3,
#   - Compute `P + w` using explicit loops and save to `P_manual`.
#   - Compute `P + w` using broadcasting (`P + w`) and save as `P_broadcast`.
#   - Print both results and confirm they are identical.
P: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
w: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])  # Shape (3,)
# Manual computation using loops
P_manual: torch.Tensor = torch.empty_like(P)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        P_manual[i, j] = P[i, j] + w[j]
# Broadcasting computation
P_broadcast: torch.Tensor = P + w
# Print results
print("Manual computation (P + w):\n", P_manual)
print("Broadcasting computation (P + w):\n", P_broadcast)
