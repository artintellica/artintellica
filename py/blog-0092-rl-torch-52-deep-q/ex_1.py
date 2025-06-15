from typing import List, Tuple
import torch
from torch import nn, Tensor


class CustomQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        dims = [state_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self.out = nn.Linear(dims[-1], action_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)
