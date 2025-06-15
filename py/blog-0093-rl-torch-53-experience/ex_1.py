from typing import List, Tuple
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

def run_dqn(use_replay: bool, num_episodes: int = 50) -> List[float]:
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0] # type: ignore
    n_actions = env.action_space.n # type: ignore

    policy_net = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=2000) if use_replay else None
    batch_size = 32
    gamma = 0.99
    target_update_freq = 50
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
            transition = (state, action, reward, next_state, done)
            if use_replay:
                memory.append(transition) # type: ignore
            else:
                # Train directly on this transition
                s, a, r, s2, d = transition
                s_v = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                a_v = torch.tensor([[a]])
                r_v = torch.tensor([r], dtype=torch.float32)
                s2_v = torch.tensor(s2, dtype=torch.float32).unsqueeze(0)
                d_v = torch.tensor([d], dtype=torch.bool)

                q_value = policy_net(s_v).gather(1, a_v).squeeze()
                with torch.no_grad():
                    next_q = target_net(s2_v).max(1)[0]
                    target = r_v + gamma * next_q * (~d_v)
                loss = nn.MSELoss()(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps_done % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if use_replay and len(memory) >= batch_size: # type: ignore
                transitions = random.sample(memory, batch_size) # type: ignore
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

rewards_with_replay = run_dqn(use_replay=True)
rewards_without_replay = run_dqn(use_replay=False)

plt.plot(rewards_with_replay, label='With Replay Buffer')
plt.plot(rewards_without_replay, label='Without Replay Buffer')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.title('Learning Curve: With vs. Without Experience Replay')
plt.show()
