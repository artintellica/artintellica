import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Any, Deque, List, Tuple
import torch.nn.functional as F

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
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class StandardDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Define replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*batch))
        states      = torch.FloatTensor(np.array(batch.state))
        actions     = torch.LongTensor(np.array(batch.action))
        rewards     = torch.FloatTensor(np.array(batch.reward))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones       = torch.FloatTensor(np.array(batch.done))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

# DuelingDQN and StandardDQN from previous code

# Hyperparameters
ENV_NAME: str = "LunarLander-v3"
GAMMA: float = 0.99
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
TARGET_UPDATE: int = 1000
REPLAY_SIZE: int = 50000
MIN_REPLAY_SIZE: int = 1000
EPSILON_START: float = 1.0
EPSILON_END: float = 0.02
EPSILON_DECAY: int = 10000
EPISODES: int = 500

def select_action(state: np.ndarray, policy_net: nn.Module, epsilon: float, action_dim: int) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = policy_net(state_t)
        return int(torch.argmax(q_vals, dim=1).item())

def train_lander(model_type: str = "double", dueling: bool = False, plot: bool = True) -> List[float]:
    env = gym.make(ENV_NAME)
    state_dim: int = env.observation_space.shape[0] # type: ignore
    action_dim: int = env.action_space.n    # type: ignore
    if dueling:
        policy_net = DuelingDQN(state_dim, action_dim)
        target_net = DuelingDQN(state_dim, action_dim)
    else:
        policy_net = StandardDQN(state_dim, action_dim)
        target_net = StandardDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay = ReplayBuffer(REPLAY_SIZE)

    steps_done = 0
    epsilon = EPSILON_START
    rewards_per_episode: List[float] = []

    state, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # Epsilon-greedy action
            epsilon = np.interp(steps_done, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward # type: ignore

            # Training step
            if len(replay) > BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                # DQN/Double DQN target calculation
                with torch.no_grad():
                    if model_type == "double":
                        # Double DQN update
                        next_q_online = policy_net(next_states)
                        next_actions = torch.argmax(next_q_online, dim=1)
                        next_q_target = target_net(next_states)
                        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        # Standard DQN update
                        next_q_target = target_net(next_states)
                        next_q_values, _ = next_q_target.max(dim=1)

                    targets = rewards + GAMMA * next_q_values * (1 - dones)

                q_values = policy_net(states)
                action_qs = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(action_qs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target net
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            steps_done += 1

        rewards_per_episode.append(total_reward)
        if (ep + 1) % 10 == 0:
            mean_reward = np.mean(rewards_per_episode[-10:])
            print(f"Episode {ep+1}, mean reward (last 10): {mean_reward:.2f}")

    env.close()
    if plot:
        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{'Dueling ' if dueling else ''}{model_type.title()} DQN on LunarLander-v2")
        plt.show()
    return rewards_per_episode

# Run & Plot
# Standard DQN
rewards_dqn = train_lander(model_type="dqn", dueling=False)
# Double DQN
rewards_double = train_lander(model_type="double", dueling=False)
# Dueling Double DQN
rewards_dueling = train_lander(model_type="double", dueling=True)

# Side-by-side plot
plt.plot(rewards_dqn, label="DQN")
plt.plot(rewards_double, label="Double DQN")
plt.plot(rewards_dueling, label="Dueling Double DQN")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title("LunarLander-v2: DQN vs Double DQN vs Dueling DQN")
plt.show()
