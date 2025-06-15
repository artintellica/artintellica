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

import gymnasium as gym
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt

# Define network, buffer, hyperparameters
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0] # type: ignore
action_dim = env.action_space.n # type: ignore

q_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
target_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
target_net.load_state_dict(q_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBufferWithIndices(5000)

BATCH_SIZE = 64
GAMMA = 0.99
TARGET_FREQ = 50
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 750

def select_action(state: np.ndarray, step: int) -> int:
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step/EPS_DECAY)
    if np.random.rand() < eps:
        return np.random.randint(action_dim)
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0)
        return int(q_net(s).argmax(1).item())

rewards_per_episode: List[float] = []
total_steps = 0
for episode in range(200):
    state, _ = env.reset()
    done = False
    ep_rew = 0.0
    while not done:
        action = select_action(state, total_steps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done) # type: ignore
        state = next_state
        ep_rew += reward # type: ignore
        total_steps += 1

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones, _ = buffer.sample_indices(BATCH_SIZE)
            s = torch.from_numpy(states).float()
            a = torch.from_numpy(actions).long().unsqueeze(1)
            r = torch.from_numpy(rewards).float().unsqueeze(1)
            ns = torch.from_numpy(next_states).float()
            d = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1)
            q = q_net(s).gather(1, a)
            with torch.no_grad():
                q_next = target_net(ns).max(1, keepdim=True)[0]
                y = r + GAMMA * q_next * (1 - d)
            loss = torch.nn.functional.mse_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % TARGET_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())
    print(f"Episode {episode + 1}: Reward = {ep_rew}")
    rewards_per_episode.append(ep_rew)

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Rewards per Episode")
plt.show()
env.close()
