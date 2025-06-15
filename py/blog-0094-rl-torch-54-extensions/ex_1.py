import torch
import torch.nn as nn
from typing import Tuple


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Scalar V(s)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # A(s, a) for each action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals  # shape: (batch, action_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_states: torch.Tensor,
    gamma: float,
    online_net: nn.Module,
    target_net: nn.Module,
) -> torch.Tensor:
    """
    Compute Double DQN targets.

    Args:
        rewards: (batch, )
        dones: (batch, )
        next_states: (batch, state_dim)
        gamma: Discount factor
        online_net: The current DQN/dueling-DQN network
        target_net: The target network

    Returns:
        Q_targets: shape (batch, )
    """
    with torch.no_grad():
        next_q_online = online_net(next_states)  # (batch, action_dim)
        next_actions = torch.argmax(next_q_online, dim=1)  # (batch, )
        next_q_target = target_net(next_states)  # (batch, action_dim)
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(
            1
        )  # (batch, )
        # Q_target = r + gamma * Q(s', a*) * (1 - done)
        Q_targets = rewards + gamma * next_q_values * (1 - dones)
    return Q_targets
