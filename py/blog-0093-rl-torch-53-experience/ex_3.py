from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def run_dqn(buffer_size: int, target_update_freq: int, num_episodes: int = 30) -> List[float]:
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0] # type: ignore
    n_actions = env.action_space.n # type: ignore

    policy_net = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=buffer_size)
    batch_size = 32
    gamma = 0.99
    all_rewards: List[float] = []
    steps_done = 0

    def select_action(state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return env.action_space.sample()
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_v)
        return q_values.argmax().item()

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state, epsilon=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward # type: ignore
            memory.append((state, action, reward, next_state, done))

            if len(memory) >= batch_size:
                transitions = random.sample(memory, batch_size)
                batch = list(zip(*transitions))
                states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
                actions = torch.tensor(batch[1], dtype=torch.long)
                rewards = torch.tensor(batch[2], dtype=torch.float32)
                next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
                dones = torch.tensor(batch[4], dtype=torch.bool)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (~dones)
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps_done % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            state = next_state
            steps_done += 1
        all_rewards.append(total_reward)
    env.close()
    return all_rewards

buffer_sizes = [1000, 5000, 10000]
target_update_freqs = [10, 50, 200]

results: List[List[float]] = []
labels: List[str] = []
for buffer_size in buffer_sizes:
    for freq in target_update_freqs:
        label = f"Buf={buffer_size},TgtUpd={freq}"
        print(f"Running config: {label}")
        rewards = run_dqn(buffer_size, freq)
        results.append(rewards)
        labels.append(label)

for rewards, label in zip(results, labels):
    plt.plot(rewards, label=label)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Replay Buffer Size and Target Update Frequency')
plt.legend()
plt.show()
