+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.5: Monte Carlo and Temporal Difference (TD) Learning"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

Reinforcement Learning isn't just about Q-learning or SARSA—at their core are powerful value estimation ideas: **Monte Carlo (MC)** and **Temporal Difference (TD)** learning. These methods let you estimate how good it is to be in a state, with different trade-offs in bias/variance and speed of learning.

In this post, you'll:

- Implement MC and TD(0) methods for estimating values in a toy MDP
- Track how estimates evolve over episodes and visualize convergence
- Use these value functions to improve policies
- Apply the ideas to the classic RL benchmark: FrozenLake-v1

---

## Mathematics: Monte Carlo vs. TD Learning

### **Monte Carlo (MC) Estimation**

Suppose you're estimating the value of state $s$: $V(s) = \mathbb{E}[G_t \mid s_t = s]$  
where $G_t$ is the **return** (sum of rewards over the episode, possibly discounted) starting from timestep $t$.

**Update Rule (after each episode):**
$$
V(s) \leftarrow V(s) + \alpha \, [G - V(s)]
$$
where $G$ is the observed return for the visit to $s$ in the episode.

- MC waits until the episode is over before updating.

---

### **TD(0) Learning**

TD methods update after **every step**, using a bootstrapped estimate:

$$
V(s_t) \leftarrow V(s_t) + \alpha \, [r_t + \gamma V(s_{t+1}) - V(s_t)]
$$

- TD(0) can learn incrementally and online.
- Trades higher bias for lower variance compared to MC.

---

## Explanation: MC and TD in Code

- **State Value Table:** Store $V(s)$ for each state $s$ (could use a NumPy array for small MDPs).
- **MC:** Track full episodes, then reward-to-go for each visited state; update $V(s)$ after trajectories.
- **TD(0):** Update $V(s)$ immediately after every transition, using $r_t + \gamma V(s_{t+1})$ as the target.
- **Tracking:** Store $V(s)$ after each episode to observe/plot convergence.
- **Policy Improvement:** With better $V(s)$, you can act greedier (e.g., always pick the action that leads to the best $V(s')$).
- **FrozenLake:** Use Gymnasium to get a challenging, stochastic MDP with visualizable states.

---

## Python Demonstrations

### Demo 1: Implement MC and TD(0) Value Updates for a Toy MDP

Let’s use our earlier 3-state MDP with random exploration.

```python
import numpy as np
import random

states = ['A', 'B', 'C']
actions = ['left', 'right']
P = {
    'A': {'left': [(1.0, 'A')], 'right': [(1.0, 'B')]},
    'B': {'left': [(0.8, 'A'), (0.2, 'C')], 'right': [(1.0, 'C')]},
    'C': {'left': [(1.0, 'B')], 'right': [(1.0, 'C')]}
}
R = {
    'A': {'left': {'A': 0.0}, 'right': {'B': 1.0}},
    'B': {'left': {'A': 0.0, 'C': 2.0}, 'right': {'C': 3.0}},
    'C': {'left': {'B': 0.0}, 'right': {'C': 0.0}}
}
def sample_next_state(s,a):
    return random.choices([ns for _,ns in P[s][a]], [p for p,_ in P[s][a]])[0]
def get_reward(s,a,s2): return R[s][a][s2]
gamma = 0.95

# Initialize value tables
V_mc = {s: 0.0 for s in states}
V_td = {s: 0.0 for s in states}
alpha = 0.1

def run_episode_MC(V, epsilon=0.3):
    # Generate an episode using epsilon-random actions
    traj = []
    s = 'A'
    for t in range(6):
        a = random.choice(actions) if random.random() < epsilon else 'right'
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        traj.append((s, a, r))
        s = s2
    # Compute MC returns and update V
    G = 0.0
    for t in reversed(range(len(traj))):
        s, a, r = traj[t]
        G = r + gamma * G
        V[s] += alpha * (G - V[s])
    return list(V.values())

def run_episode_TD(V, epsilon=0.3):
    s = 'A'
    for t in range(6):
        a = random.choice(actions) if random.random() < epsilon else 'right'
        s2 = sample_next_state(s, a)
        r = get_reward(s,a,s2)
        V[s] += alpha * (r + gamma * V[s2] - V[s])
        s = s2
    return list(V.values())
```

---

### Demo 2: Track Value Estimates Over Episodes and Plot

```python
mc_trace = []
td_trace = []
for ep in range(120):
    mc_trace.append(run_episode_MC(V_mc))
    td_trace.append(run_episode_TD(V_td))
mc_trace = np.array(mc_trace)
td_trace = np.array(td_trace)

import matplotlib.pyplot as plt
for i, s in enumerate(states):
    plt.plot(mc_trace[:,i], label=f"MC {s}")
    plt.plot(td_trace[:,i], '--', label=f"TD {s}")
plt.legend(); plt.xlabel("Episode")
plt.ylabel("Value Estimate")
plt.title("State Value Estimates: MC vs TD(0)")
plt.grid(); plt.show()
```

---

### Demo 3: Compare Convergence of MC and TD(0)

Notice the *relative speed and stability* in the plots above—TD often converges faster or is less noisy in small MDPs. Try varying $\alpha$ or $\gamma$!

---

### Demo 4: Use Value Estimates to Improve Policy and Evaluate Performance

Let’s act greedily w.r.t the estimated $V(s)$:

```python
def greedy_policy(V):
    # Always pick the action leading to the next state with highest value
    policy = {}
    for s in states:
        best_a = max(actions, key=lambda a: sum(p * V[s2] for p, s2 in P[s][a]))
        policy[s] = best_a
    return policy

best_policy_mc = greedy_policy(V_mc)
best_policy_td = greedy_policy(V_td)
print("Greedy policy from MC values:", best_policy_mc)
print("Greedy policy from TD values:", best_policy_td)
```
Now, simulate a few episodes using this policy and compute average total reward.

---

### Project Exercise: Visualize Value Function Estimates on FrozenLake-v1

Try value estimation methods on a real environment!

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n  # type: ignore
V_td = np.zeros(n_states)
gamma = 0.99
alpha = 0.08
episodes = 5000
history = []

for ep in range(episodes):
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()  # random policy
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        V_td[s] += alpha * (r + gamma * V_td[s2] - V_td[s])
        s = s2
    history.append(V_td.copy())

history = np.array(history)
# Plot value trace for first 10 states
plt.figure(figsize=(8,5))
for i in range(min(10, n_states)):
    plt.plot(history[:,i], label=f"State {i}")
plt.xlabel("Episode"); plt.ylabel("Value Estimate")
plt.legend(); plt.title("TD(0) Value Learning on FrozenLake")
plt.grid(); plt.show()

# Visualize value as an 4x4 grid (for small FrozenLake)
plt.figure(figsize=(3,3))
plt.imshow(V_td.reshape(4,4), cmap='viridis')
plt.colorbar(label="V(s)")
plt.title("Value Function (TD) for FrozenLake 4x4")
plt.show()
```

---

## Exercises

### **Exercise 1:** Implement MC and TD(0) Value Updates for a Toy MDP

- Define a simple MDP (3-state, with transitions/rewards).
- Write separate functions for MC and TD(0) value learning.

---

### **Exercise 2:** Track Value Estimates Over Episodes and Plot

- Store value tables after each episode.
- Plot how each state’s estimate evolves.

---

### **Exercise 3:** Compare Convergence of MC and TD(0)

- Use identical learning rates and random explorations.
- Overlay their value estimates to see speed and stability.

---

### **Exercise 4:** Use Value Estimates to Improve Policy and Evaluate Performance

- Generate greedy (or $\varepsilon$-greedy) policies from $V(s)$.
- Simulate total reward with improved policy and compare to random.

---

#### Project Exercise: Visualize Value Function Estimates on FrozenLake-v1

- Use random or $\varepsilon$-greedy exploration to estimate $V(s)$ for all states via TD(0).
- Visualize the final value function as a grid.

---

### **Sample Starter Code**

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# Toy MDP setup
states = ['A', 'B', 'C']
actions = ['left', 'right']
P = {'A': {'left': [(1.0, 'A')], 'right': [(1.0, 'B')]}, 'B': {'left': [(0.8, 'A'), (0.2, 'C')], 'right': [(1.0, 'C')]}, 'C': {'left': [(1.0, 'B')], 'right': [(1.0, 'C')]}}
R = {'A': {'left': {'A': 0.0}, 'right': {'B': 1.0}}, 'B': {'left': {'A': 0.0, 'C': 2.0}, 'right': {'C': 3.0}}, 'C': {'left': {'B': 0.0}, 'right': {'C': 0.0}}}

def sample_next_state(s,a):
    return random.choices([ns for _,ns in P[s][a]], [p for p,_ in P[s][a]])[0]
def get_reward(s,a,s2): return R[s][a][s2]

V_mc = {s: 0.0 for s in states}
V_td = {s: 0.0 for s in states}
gamma, alpha = 0.95, 0.12

def run_episode_MC(V):
    traj = []
    s = 'A'
    for _ in range(7):
        a = random.choice(actions)
        s2 = sample_next_state(s,a)
        r = get_reward(s,a,s2)
        traj.append((s,r))
        s = s2
    G = 0
    for t in reversed(range(len(traj))):
        s, r = traj[t]
        G = r + gamma * G
        V[s] += alpha * (G - V[s])

def run_episode_TD(V):
    s = 'A'
    for _ in range(7):
        a = random.choice(actions)
        s2 = sample_next_state(s,a)
        r = get_reward(s,a,s2)
        V[s] += alpha * (r + gamma * V[s2] - V[s])
        s = s2

# Track and plot (see above Demos)
```

---

## Conclusion

You now understand how RL can estimate the value of a state from sampled experience—by waiting for full returns (MC), or updating after each step (TD). You’ve seen the learning curves and how these methods drive practical RL agents, even in real environments like FrozenLake.

Next, you’ll connect value and policy—moving from estimates to decisions, and learn the Bellman equations that structure *all* of RL.

See you in Part 4.6!
