import torch
import torch.nn as nn
from typing import Any

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Input: state tensor of shape (batch_size, state_dim)
        Output: Q-values, tensor of shape (batch_size, n_actions)
        """
        return self.net(state)

# Example: state has 6 features, 4 possible actions
state_dim: int = 6
n_actions: int = 4
q_net = QNetwork(state_dim, n_actions)
# Batch of states
states: torch.Tensor = torch.rand(8, state_dim)
q_values: torch.Tensor = q_net(states)
print(q_values.shape)  # (8, 4)
