+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.4: Tabular Value-Based Methods—Q-Learning and SARSA"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

After seeing bandits, let’s step up to environments where actions have **consequences across time**! In gridworlds and many RL classics, the key concept is the **Q-value**: estimating the *future value* of every action in every state.

In this post, you’ll:

- Implement a Q-table for a finite environment
- Run and compare **Q-learning** and **SARSA** on Gridworld
- Visualize Q-tables as heatmaps
- Animate an agent’s path following the learned policy
- (Project) Let a Q-learning agent master Taxi-v3

---

## Mathematics: Q-Learning and SARSA

### **Q-Value Definition**

The Q-value $Q(s,a)$ is the expected sum of discounted rewards from taking action $a$ in state $s$, and then following some policy thereafter.

### **Q-Learning (Off-policy)**

Update rule:
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
\]

- $\alpha$ — learning rate
- $\gamma$ — discount factor

### **SARSA (On-policy)**

Update rule:
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma Q(s', a') - Q(s, a)\Big]
\]
where $a'$ is the actual next action taken by the agent according to its policy.

- **Q-learning** learns from the best possible next action (greedy) regardless of what the agent actually does (off-policy).
- **SARSA** learns from what the agent *actually does* (on-policy).

---

## Explanation: How the Math Connects to Code

- **Q-table**: A 2D array or dictionary storing the estimated $Q(s,a)$ for every $(s,a)$.
- **Epsilon-greedy policy**: Choose the best action most of the time, but sometimes pick randomly to explore.
- **Episode loop**: At each step, update the Q-table using either Q-learning or SARSA rules.
- **Visualization**: Use matplotlib to display the Q-values (for a gridworld, as a heatmap), and to animate learned behavior.

---

## Python Demonstrations

### Demo 1: Implement a Q-table for a Small Discrete Environment

Let’s use a tiny Gridworld with 4 states and 2 actions (LEFT, RIGHT):

```python
import numpy as np

n_states = 4
n_actions = 2
Q = np.zeros((n_states, n_actions))  # Q[state, action]

print("Initial Q-table:")
print(Q)
```

### Demo 2: Run Q-learning and SARSA on Gridworld; Compare Convergence

Let’s define a small Gridworld, reward at the end, with transitions:

```python
import random

class SimpleGridEnv:
    def __init__(self, n=4):
        self.n = n
        self.reset()
    def reset(self) -> int:
        self.s = 0
        return self.s
    def step(self, a: int) -> tuple[int, float, bool]:
        # a: 0=LEFT, 1=RIGHT
        if a == 1:
            self.s += 1
        else:
            self.s -= 1
        self.s = np.clip(self.s, 0, self.n-1)
        reward = 1.0 if self.s == self.n-1 else 0.0
        done = self.s == self.n-1
        return self.s, reward, done

def epsilon_greedy(Q, s, eps=0.2):
    if random.random() < eps:
        return random.choice(range(n_actions))
    return int(np.argmax(Q[s]))

def train_q(env, Q, episodes=250, alpha=0.1, gamma=0.9, eps=0.2):
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.
        while not done:
            a = epsilon_greedy(Q, s, eps)
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])  # Q-learning update
            s = s2
            total += r
        returns.append(total)
    return np.array(returns)

def train_sarsa(env, Q, episodes=250, alpha=0.1, gamma=0.9, eps=0.2):
    returns = []
    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps)
        done = False
        total = 0.
        while not done:
            s2, r, done = env.step(a)
            a2 = epsilon_greedy(Q, s2, eps)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])  # SARSA update
            s = s2
            a = a2
            total += r
        returns.append(total)
    return np.array(returns)

env = SimpleGridEnv(n_states)
Q1 = np.zeros((n_states, n_actions))
Q2 = np.zeros((n_states, n_actions))
rets1 = train_q(env, Q1)
rets2 = train_sarsa(env, Q2)

import matplotlib.pyplot as plt
plt.plot(np.cumsum(rets1) / (np.arange(len(rets1))+1), label="Q-Learning")
plt.plot(np.cumsum(rets2) / (np.arange(len(rets2))+1), label="SARSA")
plt.ylabel("Mean Return"); plt.xlabel("Episode")
plt.legend(); plt.title("Q-Learning vs SARSA (Gridworld)")
plt.grid(); plt.show()
```

---

### Demo 3: Visualize the Learned Q-table as a Heatmap

```python
from matplotlib import pyplot as plt

plt.imshow(Q1, cmap='cool', interpolation='nearest')
plt.colorbar(label="Q-value")
plt.title("Q-table for Q-learning (States x Actions)")
plt.xlabel("Action (0=Left, 1=Right)")
plt.ylabel("State")
plt.show()
```

---

### Demo 4: Animate Agent’s Trajectory Using the Learned Policy

```python
import time

def run_policy(env, Q, max_steps=10, delay=0.4) -> None:
    s = env.reset()
    traj = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s2, r, done = env.step(a)
        traj.append(s2)
        print(f"State: {s} -> Action: {a} -> State: {s2} | Reward: {r}")
        s = s2
        if done:
            break
        time.sleep(delay)
    print("Trajectory:", traj)

print("Animating Q-learning policy:")
run_policy(env, Q1)
```

---

### Project Exercise: Train a Q-Learning Agent on Taxi-v3 and Report Average Episode Length

Taxi-v3 is a classic Gym environment—use Q-learning to learn its optimal policy!

```python
import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode="ansi")
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))
episodes = 1800
alpha = 0.1
gamma = 0.98
eps = 0.15

lengths = []
for ep in range(episodes):
    s, _ = env.reset()
    done = False
    count = 0
    while not done:
        if np.random.rand() < eps:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s = s2
        count += 1
    lengths.append(count)

import matplotlib.pyplot as plt
plt.plot(np.convolve(lengths, np.ones(50)/50, mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Taxi-v3: Q-Learning Episode Length (lower is better)")
plt.show()
print("Mean episode length (last 100 episodes):", np.mean(lengths[-100:]))
```

---

## Exercises

### **Exercise 1:** Implement a Q-table for a Small Discrete Environment

- Create a 2D array or dictionary `Q[s, a]` with zeros.
- Print it to check shape and initial values.

---

### **Exercise 2:** Run Q-learning and SARSA on Gridworld, Compare Convergence

- Train Q-learning and SARSA separately on the same gridworld.
- Plot mean reward over time for both methods.

---

### **Exercise 3:** Visualize the Learned Q-table as a Heatmap

- After learning, plot `Q` (states by actions) as an imshow/heatmap.
- Label axes for clarity.

---

### **Exercise 4:** Animate Agent’s Trajectory Using the Learned Policy

- For a trained policy (greedy from Q-table), print or animate state/action transitions from a start state.

---

### **Project Exercise:** Train a Q-learning Agent on Taxi-v3 and Report Average Episode Length

- Use Gymnasium’s `Taxi-v3` environment and Q-learning. Train for 1–2k episodes.
- Plot moving average of episode lengths.
- Report average over the last 100 episodes.

---

### **Sample Starter Code for Exercises**

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# Exercise 1
n_states, n_actions = 4, 2
Q = np.zeros((n_states, n_actions))
print(Q)

# Exercise 2/3
# (See above code!)
# Use SimpleGridEnv, epsilon_greedy, train_q, train_sarsa
# Plot learning curves and Q-table heatmap

# Exercise 4
env = SimpleGridEnv(n_states)
run_policy(env, Q1)
```

---

## Conclusion

You now know how to:
- Represent and learn Q-values in tabular environments
- Run and compare Q-learning (off-policy) and SARSA (on-policy)
- Visualize Q-tables and agent behavior

Next, you’ll see how random sampling and partial returns let us estimate value without knowing the environment—*Monte Carlo* and *TD* learning.

See you in Part 4.5!
