import torch
import matplotlib.pyplot as plt

# - Use the same `a` and `b` as Exercise 2.
# - Calculate the cosine similarity using both the formula and
#   `torch.nn.functional.cosine_similarity`.
a: torch.Tensor = torch.tensor([1.0, 7.0, 2.0, 5.0])
b: torch.Tensor = torch.tensor([5.0, 1.0, 2.0, -1.0])
# Manual cosine similarity calculation
a_norm: torch.Tensor = torch.norm(a)
b_norm: torch.Tensor = torch.norm(b)
cos_sim_manual: torch.Tensor = torch.dot(a, b) / (a_norm * b_norm)
print("a:", a)
print("b:", b)
print("Cosine similarity (manual):", cos_sim_manual.item())
# Built-in cosine similarity calculation
cos_sim_builtin: torch.Tensor = torch.nn.functional.cosine_similarity(
    a.unsqueeze(0), b.unsqueeze(0)
)
print("Cosine similarity (PyTorch builtin, batch):", cos_sim_builtin.item())
