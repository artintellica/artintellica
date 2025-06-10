import torch

A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

try:
    bad_vec: torch.Tensor = torch.tensor([1.0, 2.0])
    # A has shape (3,3); bad_vec has shape (2,). Not broadcastable!
    res = A + bad_vec
except RuntimeError as e:
    print("Broadcasting error:", e)

# Fix: Use a vector of shape (3,) or reshape for broadcasting compatibility
good_vec: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
fixed_res: torch.Tensor = A + good_vec
print("Fixed result (A + good_vec):\n", fixed_res)
