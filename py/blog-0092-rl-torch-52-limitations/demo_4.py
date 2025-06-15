import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # Shape: [batch, action_dim]


import random
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
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

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
GAMMA: float = 0.99
BATCH_SIZE: int = 64
REPLAY_SIZE: int = 10000
LEARNING_RATE: float = 1e-3
TARGET_UPDATE_FREQ: int = 100
EPS_START: float = 1.0
EPS_END: float = 0.01
EPS_DECAY: int = 500  # steps

# Initialize environment and networks
env = gym.make("CartPole-v1")
state_dim: int = env.observation_space.shape[0] if env.observation_space.shape is not None else 0
action_dim: int = env.action_space.n # type: ignore

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_SIZE)

# Epsilon-greedy policy
def select_action(state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = q_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

# Training loop
num_episodes: int = 50
episode_rewards: List[float] = []
steps_done: int = 0

for episode in range(num_episodes):
    state, _ = env.reset(seed=None)
    episode_reward: float = 0.0
    done = False
    while not done:
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done) # type: ignore
        state = next_state
        episode_reward += reward # type: ignore
        steps_done += 1

        # Train step
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states_t = torch.from_numpy(states).float()
            actions_t = torch.from_numpy(actions).long().unsqueeze(1)
            rewards_t = torch.from_numpy(rewards).float().unsqueeze(1)
            next_states_t = torch.from_numpy(next_states).float()
            dones_t = torch.from_numpy(dones.astype(np.float32)).float().unsqueeze(1)

            # Q(s_t, a)
            q_values = q_net(states_t).gather(1, actions_t)
            # Compute target Q-values
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                targets = rewards_t + GAMMA * max_next_q * (1 - dones_t)

            # Loss
            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target net
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

    episode_rewards.append(episode_reward)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training on CartPole-v1")
plt.show()

# Save model weights
torch.save(q_net.state_dict(), "dqn_cartpole.pth")

# To load:
q_net_loaded = QNetwork(state_dim, action_dim)
q_net_loaded.load_state_dict(torch.load("dqn_cartpole.pth", map_location=torch.device("cpu")))
q_net_loaded.eval()
