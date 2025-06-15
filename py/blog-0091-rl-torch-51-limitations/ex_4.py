import torch
import torch.nn as nn


class BigQNetwork(nn.Module):
    def __init__(
        self, state_dim: int = 8, n_actions: int = 6, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


qnet = BigQNetwork()
n_params = sum(p.numel() for p in qnet.parameters())
print(f"Q-network has {n_params:,} parameters (trainable)")
# Example usage: for a batch of 10 states
state_batch = torch.rand(10, 8)
q_outputs = qnet(state_batch)
print(f"Output shape: {q_outputs.shape}")  # (10, 6)
