+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.4: Extensions—Double DQN and Dueling DQN"
author = "Artintellica"
date = "2024-06-15"
+++

## Introduction

Advancements in Deep Q-Learning have brought us remarkable performance in
complex reinforcement learning (RL) environments. However, standard DQN (Deep
Q-Network) displays some well-known limitations, including overestimating action
values and inefficient representation of state-value and action-advantages. In
this blog post, we introduce two essential extensions that address these
problems—**Double DQN** and **Dueling DQN**. We'll cover the mathematical
motivation, provide practical PyTorch code, and guide you through hands-on
exercises to train, compare, and visualize these improved algorithms on the
[LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
environment.

---

## Mathematical Overview

### Q-Learning Recap

Q-Learning seeks to estimate the optimal action-value function:

$$
Q^*(s, a) = \max_\pi \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a, \pi \right]
$$

where $r_t$ is the reward at time $t$, $\gamma$ is the discount factor, and
$\pi$ is a policy.

**DQN** uses deep neural networks as a function approximator for
$Q(s, a;\,\theta)$ and mini-batch Q-learning updates:

$$
y^\text{DQN} = r + \gamma \max_{a'} Q_\text{target}(s', a';\,\theta^-)
$$

where $\theta^-$ are the parameters of the target network.

### Double DQN: Reducing Overestimation Bias

DQN tends to overestimate action values due to the **max operator** in the
target. Double DQN fixes this using separate networks for selection and
evaluation:

$$
y^\text{Double DQN} = r + \gamma Q_\text{target}\big(s', \underset{a'}{\arg\max}\ Q_\text{online}(s', a';\,\theta),\ \theta^- \big)
$$

- $\theta$: parameters of the online network
- $\theta^-$: parameters of the target network

Here, **action selection** uses the online network, and **action evaluation**
uses the target.

### Dueling DQN: Separating State-Value and Advantage

Dueling DQN proposes a new architecture:

$$
Q(s, a;\,\theta, \alpha, \beta) = V(s;\,\theta, \beta) + \left( A(s, a;\,\theta, \alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\,\theta, \alpha) \right)
$$

- $V(s)$: Value function of state
- $A(s, a)$: Advantage function for action $a$ in state $s$

This separation helps the network learn which states are valuable irrespective
of the action taken, improving training speed and performance in some
environments.

---

## Connecting the Math to the Code

- **Double DQN**:
  - When calculating Q-learning targets, select the greedy action using the
    **online** (current) network, but evaluate its Q-value using the **target**
    network.
- **Dueling DQN**:
  - Design your Q-network such that, after some shared layers, you create two
    streams:
    - One outputs the scalar state value $V(s)$.
    - The other outputs the advantages $A(s, a)$ for all actions.
  - These are recombined as per the formula above to produce the Q-values for
    each action.

We'll provide full PyTorch implementations as hands-on Python demos and
exercises below.

---

## Python Demos

Let's start with the essential building blocks. These will be reused in the
exercises that follow.

### Demo 1: Dueling DQN Network Class

```python
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
```

---

## Exercises

### Exercise 1: Extend your DQN to Double DQN and Compare Performance

**Description:** Modify your existing DQN code to implement Double DQN.
Specifically, during the target calculation, use the online network to select
the next action, but use the target network to evaluate its Q-value.

**Full Double DQN target computation with fully typed PyTorch code:**

```python
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
    target_net: nn.Module
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
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (batch, )
        # Q_target = r + gamma * Q(s', a*) * (1 - done)
        Q_targets = rewards + gamma * next_q_values * (1 - dones)
    return Q_targets
```

---

### Exercise 2: Implement Dueling Network Architecture in PyTorch

**Description:** Write a complete Dueling DQN network class (if you haven't done
so in the demo already!), and compare its outputs with a standard MLP DQN model
on random input.

**Full code:**

```python
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

# Test on random input
if __name__ == "__main__":
    state_dim: int = 8   # for LunarLander-v2
    action_dim: int = 4
    x: torch.Tensor = torch.randn(5, state_dim)

    dueling_net = DuelingDQN(state_dim, action_dim)
    standard_net = StandardDQN(state_dim, action_dim)
    print("Dueling DQN output:", dueling_net(x))
    print("Standard DQN output:", standard_net(x))
```

---

### Exercise 3: Train Both Models on LunarLander-v2 and Compare Results

**Description:** Train both DQN/Double DQN and Dueling DQN agents on
[LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/).
Track and plot average episode rewards for each.

**Full code for a minimal training loop (assumes experience replay
implementation):**

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Any, Deque, List, Tuple

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
ENV_NAME: str = "LunarLander-v2"
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
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
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
            total_reward += reward

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
```

---

### Exercise 4: Visualize Q-Value Distributions During Training

**Description:** During or after training, visualize the distribution
(histogram) of Q-values predicted by your DQN/Double DQN/Dueling DQN for a set
of states. Repeat periodically to observe stability and learning
characteristics.

**Full code:**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_q_distributions(
    states: np.ndarray,
    model: nn.Module,
    action_dim: int,
    title: str = "Q-value distribution"
) -> None:
    """
    Plot histogram of Q-values for a batch of states.

    Args:
        states: np.ndarray of shape (batch_size, state_dim)
        model: Q-network
        action_dim: number of actions
        title: plot title
    """
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        q_vals = model(state_tensor)  # (batch, action_dim)
        q_vals_flat = q_vals.cpu().numpy().flatten()
        plt.hist(q_vals_flat, bins=30, alpha=0.7)
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.show()
    model.train()

# Example usage, after training any agent:
sample_states = np.random.randn(500, 8)  # or better: sample from the replay buffer
plot_q_distributions(sample_states, trained_policy_net, action_dim=4, title="Trained Double DQN Q-value Distribution")
```

---

### Project Exercise: Plot Side-by-Side Learning Curves

**Description:** Aggregate the reward curves obtained after training DQN, Double
DQN, and Dueling DQN agents, and plot them in the same graph for side-by-side
comparison.

**Full code:**

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_side_by_side_curves(
    curves: List[List[float]],
    labels: List[str],
    window: int = 10,
    title: str = "Learning Curves"
) -> None:
    plt.figure(figsize=(10,6))
    for curve, label in zip(curves, labels):
        if window > 1:
            rolling = np.convolve(curve, np.ones(window)/window, mode='valid')
            plt.plot(rolling, label=label)
        else:
            plt.plot(curve, label=label)
    plt.xlabel("Episode")
    plt.ylabel(f"Mean Reward (window={window})")
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage:
# plot_side_by_side_curves([rewards_dqn, rewards_double, rewards_dueling], ["DQN", "Double DQN", "Dueling DQN"])
```

---

## Conclusion

You've learned how Double DQN reduces overestimation bias by splitting action
selection and evaluation, and how Dueling DQN architectures represent value and
advantage functions separately for more robust learning. With PyTorch code,
you've extended vanilla DQN, built dueling architectures, trained the agents on
LunarLander-v2, and visualized both learning progress and internal Q-value
statistics.

**Next:** In the following posts, we'll introduce policy gradient methods and
actor-critic algorithms—taking RL one step closer toward solving real-world
sequential decision-making tasks!

---

**Try out the exercises with your own enhancements, and share your learning
curves in the comments!**
