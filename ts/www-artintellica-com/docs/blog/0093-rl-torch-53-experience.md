+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.3: Experience Replay and Target Networks"
author = "Artintellica"
date = "2024-06-15"
+++

## Introduction

In the world of deep reinforcement learning (RL), scaling value-based algorithms
like Q-learning beyond small, tabular environments presents unique challenges.
As you may have experienced, directly plugging neural networks into the RL loop
often leads to instability and divergence. Two foundational ideas—**experience
replay** and the use of a **target network**—help address these issues and form
the backbone of algorithms like Deep Q-Networks (DQN).

In this post, you'll learn the mathematical motivation for these techniques, see
how they're implemented in PyTorch, and observe their profound effect on agent
stability and learning efficiency. We'll wrap up every section with hands-on
exercises and full-typed code so you can practice and experiment on your own.

---

## Overview of Mathematical Concepts

### The Problem: Instability in Deep Q-Learning

Traditional Q-learning incrementally updates a Q-table using the Bellman
equation:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]
$$

However, when $Q$ is parameterized by a neural network $Q_\theta(s, a)$,
directly updating with new samples causes two major issues:

1. **Highly Correlated Data:** Sequential states sampled from the environment
   are not independent and identically distributed (i.i.d.), breaking
   assumptions of standard supervised learning.

2. **Unstable Targets:** The target $r + \gamma \max_{a'} Q_\theta(s', a')$ is a
   moving target because the network is learning both input and output, leading
   to divergence or oscillations.

### Experience Replay

**Experience Replay** addresses data correlation. At each time step, transitions
$(s, a, r, s', done)$ are stored in a buffer $\mathcal{D}$. Instead of training
immediately on the latest experience, we sample **mini-batches** randomly from
$\mathcal{D}$ to update the Q-network.

Mathematically, this changes the loss from:

$$
\mathcal{L}(\theta) = \left[ r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a) \right]^2
$$

to

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\Bigg[\left[ r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a) \right]^2\Bigg]
$$

This encourages more stable and efficient learning by breaking the correlation.

### Target Network

A **Target Network** enhances stability by decoupling the target calculation
from rapidly changing network parameters. Instead of using the latest parameters
for both action selection and target evaluation, we use a periodically updated
copy $Q_{\theta^-}(s, a)$ only for target computation:

$$
\text{Target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

where $\theta^-$ is updated to match $\theta$ every $N$ steps.

### Key Terms

- $\mathcal{D}$: Replay memory/buffer holding past transitions.
- $Q_\theta(s, a)$: The Q-value predicted by the current network.
- $Q_{\theta^-}(s, a)$: Q-value predicted by the target network.
- $\gamma$: Discount factor.
- $\alpha$: Learning rate.
- Mini-batch: Random sample of transitions for one gradient update.

---

## Connecting the Math to the Code

**In code:**

- The **experience replay buffer** is often implemented as a Python list or a
  ring buffer (deque), supporting efficient addition and random sampling.
- The **target network** is a deep copy of the main Q-network, updated by
  copying weights at regular intervals
  (`target_net.load_state_dict(policy_net.state_dict())` in PyTorch).
- During training, we sample a mini-batch of transitions from the replay buffer,
  calculate the target Q-values using the target network, and update the policy
  network via gradient descent.

Below, you'll find demos and exercises that show how to implement and study
these ideas with PyTorch.

---

## Python Demos

### Demo 1: Minimal Experience Replay Buffer

```python
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
```

### Demo 2: Target Network Update in PyTorch

```python
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
```

---

## Python Exercises

### Exercise 1: Implement a Target Network for Stable Q-value Updates

**Description:**  
Given a DQN agent with a `policy_net` (main Q-network), implement a target
network that is updated every fixed number of steps (`target_update_freq`). Use
`env='CartPole-v1'` for quick testing.

**Full Code:**

```python
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import random
from collections import deque

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Hyperparameters
batch_size = 32
gamma = 0.99
lr = 1e-3
target_update_freq = 100

# Environment and networks
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = QNetwork(obs_dim, n_actions)
target_net = QNetwork(obs_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = deque(maxlen=5000)

def select_action(state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return env.action_space.sample()
    state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_v)
    return q_values.argmax().item()

num_episodes = 10
steps_done = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = select_action(state, epsilon=0.1)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(memory) >= batch_size:
            transitions = random.sample(memory, batch_size)
            batch = list(zip(*transitions))
            states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
            actions = torch.tensor(batch[1], dtype=torch.long)
            rewards = torch.tensor(batch[2], dtype=torch.float32)
            next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
            dones = torch.tensor(batch[4], dtype=torch.bool)

            # Compute Q(s_t, a)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Compute target Q-values using the target network
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (~dones)

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

env.close()
print("Finished Exercise 1.")
```

---

### Exercise 2: Compare Learning With and Without Experience Replay

**Description:**  
Train two DQN agents on CartPole-v1: one uses experience replay, the other
trains on immediate transitions (no replay buffer). Compare performance (total
reward per episode).

**Full Code:**

```python
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import gym
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
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

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
            total_reward += reward
            transition = (state, action, reward, next_state, done)
            if use_replay:
                memory.append(transition)
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

            if use_replay and len(memory) >= batch_size:
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

rewards_with_replay = run_dqn(use_replay=True)
rewards_without_replay = run_dqn(use_replay=False)

plt.plot(rewards_with_replay, label='With Replay Buffer')
plt.plot(rewards_without_replay, label='Without Replay Buffer')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.title('Learning Curve: With vs. Without Experience Replay')
plt.show()
```

---

### Exercise 3: Tune Replay Buffer Size and Target Update Frequency

**Description:**  
Try different replay buffer sizes (`1000`, `5000`, `10000`) and target update
frequencies (`10`, `50`, `200`) for DQN on CartPole-v1. Plot and compare
learning curves.

**Full Code:**

```python
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import gym
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
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

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
            total_reward += reward
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
```

---

### Exercise 4: Analyze Training Stability Through Reward Variance Plots

**Description:**  
Plot the running variance of episode rewards for a DQN agent (e.g., with and
without replay buffer) to visualize and compare training stability.

**Full Code:**

```python
from typing import List
import numpy as np
import matplotlib.pyplot as plt

def running_variance(data: List[float], window_size: int = 10) -> np.ndarray:
    var_list = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        var_list.append(np.var(window))
    return np.array(var_list)

# Assume rewards_with_replay and rewards_without_replay from Exercise 2
window_size = 10
var_with_replay = running_variance(rewards_with_replay, window_size)
var_without_replay = running_variance(rewards_without_replay, window_size)

plt.plot(var_with_replay, label='With Replay Buffer')
plt.plot(var_without_replay, label='Without Replay Buffer')
plt.xlabel('Episode')
plt.ylabel('Reward Variance (Window size = 10)')
plt.title('Training Stability: Reward Variance')
plt.legend()
plt.show()
```

---

## Conclusion

Both **experience replay** and **target networks** are critical tools for
stabilizing and improving the efficiency of DQN and related algorithms. Through
the exercises, you should observe that these ideas greatly smooth learning and
make deep RL practical. Experiment with buffer sizes and update frequencies—and
try the reward variance tool to make your own stability claims concrete!

In the next post, we'll dive into advanced extensions such as Double DQN and
Dueling Networks, which push deep value-based RL even further.

**Happy experimenting, and see you in the next lesson!**
