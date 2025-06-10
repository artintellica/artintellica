import torch

D: torch.Tensor = torch.tensor([[1.0, 2.0]])
E: torch.Tensor = torch.tensor([[3.0, 4.0]])
try:
    bad_result = D @ E
except RuntimeError as err:
    print("Shape mismatch error:", err)

# Fix: Transpose E
fixed_result = D @ E.T
print("Fixed result (D @ E.T):", fixed_result)
