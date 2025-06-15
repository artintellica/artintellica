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


import numpy as np
import random
from typing import List, Tuple


class ReplayBufferWithIndices:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample_indices(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, indices

    def __len__(self) -> int:
        return len(self.buffer)
