from typing import Tuple, List, Any
import random

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[Any, ...]] = []
        self.position = 0

    def push(self, transition: Tuple[Any, ...]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Create two networks
obs_dim = 4
action_dim = 2
policy_net = QNetwork(obs_dim, action_dim)
target_net = QNetwork(obs_dim, action_dim)

# Copy parameters from policy_net to target_net
target_net.load_state_dict(policy_net.state_dict())
