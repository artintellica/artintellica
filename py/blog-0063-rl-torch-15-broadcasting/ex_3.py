import torch

# - Intentionally try to broadcast a $(5,3)$ matrix with a vector of shape $(2,)$.
# - Observe and print the error message.
# - Fix the shapes so that broadcasting works (e.g., use a vector of shape $(3,)$
#   or $(5,1)$).
A: torch.Tensor = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]
)
try:
    bad_vec: torch.Tensor = torch.tensor([1.0, 2.0])  # Shape (2,)
    # A has shape (5,3); bad_vec has shape (2,). Not broadcastable!
    res = A + bad_vec
except RuntimeError as e:
    print("Broadcasting error:", e)
# Fix: Use a vector of shape (3,) or reshape for broadcasting compatibility
good_vec: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])  # Shape (3,)
fixed_res: torch.Tensor = A + good_vec
print("Fixed result (A + good_vec):\n", fixed_res)
# Alternatively, we could reshape the vector to (2, 1) to make it broadcastable
# good_vec_reshaped: torch.Tensor = bad_vec.view(2, 1)  # Shape (2, 1)
# fixed_res_reshaped: torch.Tensor = A + good_vec_reshaped
# print("Fixed result with reshaped vector (A + good_vec_reshaped):\n", fixed_res_reshaped)
