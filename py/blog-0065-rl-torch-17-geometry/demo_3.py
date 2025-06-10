import torch

a: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
b: torch.Tensor = torch.tensor([4.0, 6.0, 8.0])
print("a:", a)
print("b:", b)

# Cosine similarity with a manual formula
a_norm: torch.Tensor = torch.norm(a)
b_norm: torch.Tensor = torch.norm(b)
cos_sim: torch.Tensor = torch.dot(a, b) / (a_norm * b_norm)
print("Cosine similarity between a and b:", cos_sim.item())

# Built-in version for batches
cos_sim_builtin: torch.Tensor = torch.nn.functional.cosine_similarity(
    a.unsqueeze(0), b.unsqueeze(0)
)
print("Cosine similarity (PyTorch builtin, batch):", cos_sim_builtin.item())
