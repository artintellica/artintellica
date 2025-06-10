import torch

d: torch.Tensor = torch.tensor([1, 3], dtype=torch.float32)
e: torch.Tensor = torch.tensor([2, 1], dtype=torch.float32)
sum_de: torch.Tensor = d + e

scaled_sum_de: torch.Tensor = sum_de * 0.5
