+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.2: Deep Q-Networks (DQN): Concepts and PyTorch Implementation"
author = "Artintellica"
date = "2024-06-15"
+++

# Learn Reinforcement Learning with PyTorch, Part 5.2: Deep Q-Networks (DQN): Concepts and PyTorch Implementation

---

## Introduction

So far in our series, we’ve explored reinforcement learning (RL) using tabular methods—explicitly storing and updating a value for every state–action pair. But the real world is far more complex: many problems have huge or continuous state spaces where tabular methods simply cannot scale. Enter Deep Q-Networks (DQN), the game-changing approach that brought deep learning to RL. This post will introduce you to DQN concepts and walk you through a full working PyTorch implementation, using OpenAI Gym’s classic **CartPole-v1** environment as our running example.

By the end, you’ll have built a neural Q-network, implemented experience replay, trained your own agent, and learned to tune and save your models. All code is hands-on, fully typed, and annotated for understanding.

---

## Mathematical Concepts Overview

Let’s review and extend the math underlying DQN.

### 1. The Q-Function and Bellman Equation

Recall that the **action-value function** (Q-function) for a state–action pair $(s, a)$ under policy $\pi$, is:

$$
Q^\pi(s, a) = \mathbb{E}\left[ G_t \mid s_t = s,\, a_t = a,\, \pi \right]
$$

where
- $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$
- $\gamma$ is the discount factor, $0 \leq \gamma < 1$

**The Bellman optimality equation for Q* (the optimal Q-function):**

$$
Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a')\ \Big|\, s, a \right]
$$

### 2. Deep Q-Networks

In DQN, we **approximate** $Q^*(s,a)$ with a neural network parameterized by $\theta$. The network takes an observation (the state) as input and outputs Q-values for every possible action.

- **Input:** state $s$
- **Output:** $Q(s, a; \theta), \ \forall a \in \mathcal{A}$

The goal: minimize the **mean squared error (MSE)** between the current Q-value and the Bellman target for experience $(s, a, r, s', done)$:

$$
\text{Target:}\quad y = 
\begin{cases}
  r & \text{if done (terminal)}\\
  r + \gamma \max_{a'} Q(s', a';\ \theta^{-}) & \text{if not done}
\end{cases}
$$

where $\theta^{-}$ are the parameters of a "target network," usually a slowly updated snapshot of the main Q-network.

**DQN Learning Objective:**

$$
L(\theta) = \mathbb{E}_{(s, a, r, s', done) \sim \mathcal{D}} \left[
    \Big(Q(s,a;\theta) - y\Big)^2
\right]
$$

where $\mathcal{D}$ is a **replay buffer** (experience memory).

### 3. Experience Replay

To break correlations and improve sample efficiency, DQN samples batches **randomly** from a replay buffer of recent experiences instead of updating from only the latest transition.

### 4. Target Networks

A separate, periodically updated copy of the Q-network ($\theta^-$) is used to compute targets $y$, stabilizing training by reducing target oscillations.

---

## From Math to Code: Building DQN Step by Step

1. **Q-Network:** A standard MLP (multi-layer perceptron) that estimates $Q(s,a)$ from input states.
2. **Replay Buffer:** Stores $(s,a,r,s',done)$ transitions; sample minibatches for updates.
3. **Training Loop:**
    - For each time step:
        - Select action (epsilon-greedy: random with probability $\epsilon$, else greedy)
        - Step environment, collect $(s,a,r,s',done)$, add to replay buffer
        - Sample batch, calculate targets, calculate MSE loss, backpropagate
    - Occasionally update target network

---

## Python Demos

Let's walk through each core DQN component, with explanations and type hints.

---

### Demo 1: Build a Q-network (`nn.Module`) for CartPole-v1

```python
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
```

---

### Demo 2: Implement an Experience Replay Buffer

```python
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

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### Demo 3: Complete DQN Agent and Training Loop for CartPole-v1

```python
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
state_dim: int = env.observation_space.shape[0]
action_dim: int = env.action_space.n

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
num_episodes: int = 300
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
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
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
```

---

### Demo 4: Save and Load a Trained DQN Model

```python
# Save model weights
torch.save(q_net.state_dict(), "dqn_cartpole.pth")

# To load:
q_net_loaded = QNetwork(state_dim, action_dim)
q_net_loaded.load_state_dict(torch.load("dqn_cartpole.pth", map_location=torch.device("cpu")))
q_net_loaded.eval()
```

---

## Python Exercises

Let's deepen your understanding with concrete coding challenges.

---

### **Exercise 1 — Build a Q-network using `nn.Module` for CartPole-v1**

_Define a class `QNetwork` as above, but allow the number of hidden layers and units per layer to be specified dynamically._

```python
from typing import List, Tuple
import torch
from torch import nn, Tensor

class CustomQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        dims = [state_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))]
        )
        self.out = nn.Linear(dims[-1], action_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)
```

---

### **Exercise 2 — Implement an experience replay buffer from scratch**

_Implement a class `ReplayBuffer` as shown, but add a method `.sample_indices(batch_size)` that returns both the sampled experiences and their indices in the buffer (useful for prioritized replay)._

```python
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

    def sample_indices(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]
    ]:
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, indices

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### **Exercise 3 — Train a DQN agent and plot episode rewards**

_Use your Q-network and replay buffer to train an agent on CartPole-v1. Keep track of rewards per episode and plot them with Matplotlib._

```python
import gymnasium as gym
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt

# Define network, buffer, hyperparameters
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

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
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_rew += reward
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
    rewards_per_episode.append(ep_rew)

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Rewards per Episode")
plt.show()
env.close()
```

---

### **Exercise 4 — Save and load trained DQN models**

_Use `torch.save` and `torch.load` to serialize and deserialize your trained DQN model's parameters._

```python
# Saving
torch.save(q_net.state_dict(), "custom_dqn.pth")

# Loading
loaded_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
loaded_net.load_state_dict(torch.load("custom_dqn.pth"))
loaded_net.eval()
```

---

### **Project Exercise — Experiment with different network sizes and plot curves**

_Train and compare DQNs with one vs. two hidden layers, 64 vs. 256 units. Plot learning curves for all four variants on the same chart._

```python
configs = [
    ([64], "1x64"),
    ([256], "1x256"),
    ([64, 64], "2x64"),
    ([256, 256], "2x256")
]
env = gym.make("CartPole-v1")
all_rewards: List[List[float]] = []

for hidden_layers, label in configs:
    q_net = CustomQNetwork(state_dim, action_dim, hidden_dims=hidden_layers)
    target_net = CustomQNetwork(state_dim, action_dim, hidden_dims=hidden_layers)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBufferWithIndices(5000)
    rewards = []
    total_steps = 0
    for episode in range(100):
        state, _ = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            eps = EPS_END + (EPS_START-EPS_END)*np.exp(-total_steps/EPS_DECAY)
            action = np.random.randint(action_dim) if np.random.rand() < eps else \
                int(q_net(torch.from_numpy(state).float().unsqueeze(0)).argmax(1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_rew += reward
            total_steps += 1
            if len(buffer) >= BATCH_SIZE:
                s, a, r, ns, d, _ = buffer.sample_indices(BATCH_SIZE)
                s = torch.from_numpy(s).float()
                a = torch.from_numpy(a).long().unsqueeze(1)
                r = torch.from_numpy(r).float().unsqueeze(1)
                ns = torch.from_numpy(ns).float()
                d = torch.from_numpy(d.astype(np.float32)).unsqueeze(1)
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
        rewards.append(ep_rew)
    all_rewards.append((label, rewards))

# Plotting
plt.figure(figsize=(10,6))
for label, rewards in all_rewards:
    plt.plot(rewards, label=label)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("DQN: Effect of Network Size on CartPole-v1")
plt.show()
env.close()
```

---

## Conclusion

Deep Q-Networks combine the strong function-approximation capability of neural networks with classic Q-learning, enabling RL agents to handle large/continuous state spaces. The essential tricks—experience replay, target networks—are all now in your hands, and you’ve coded a fully working DQN for the classic CartPole problem.

Don’t forget: you can always tweak the architecture, batch size, epsilon schedule and more to see how your agent learns. In the next lesson, we’ll push DQN further with modern extensions like Double and Dueling DQN. Happy experimenting!
