import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits


# Instantiate and print
model: SimpleNet = SimpleNet(2, 8, 2)
print(model)
