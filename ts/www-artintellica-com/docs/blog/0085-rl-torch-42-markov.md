+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.2: Markov Decision Processes—States, Actions, Rewards, Policies"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

All of reinforcement learning is built on the formal concept of a **Markov
Decision Process (MDP)**. MDPs provide the mathematical framework underlying RL
algorithms—from basic tabular Q-learning to modern deep RL.

In this post, you will:

- Define the components and math of an MDP
- Implement a custom MDP as Python code
- Simulate random and simple policies, measuring their performance
- Visualize MDP state transitions as a graph
- Lay the groundwork for value functions and policy improvement

---

## Mathematics: Markov Decision Processes

An **MDP** is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $S$ — set of **states**: $s \in \mathcal{S}$
- $A$ — set of **actions**: $a \in \mathcal{A}$
- $P$ — **transition probabilities**: $P(s'|s,a)$, probability of moving to $s'$
  from $s$ when taking action $a$
- $R$ — **reward function**: $R(s,a)$ or $R(s,a,s')$, expected immediate reward
- $\gamma$ — **discount factor**: $0 \leq \gamma \leq 1$

**Agent-Environment Loop:** At each timestep $t$: $ s*t \xrightarrow{a_t} (r_t,
s*{t+1}) $ The agent picks action $a_t$ in state $s_t$, the environment returns
reward $r_t$ and the next state $s_{t+1}$ according to $P$ and $R$.

**Policy:** $\pi(a|s)$ gives the probability the agent chooses action $a$ in
state $s$.

**Expected Return:** $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$

---

## Explanation: How the Math Connects to Code

- **States/Actions:** In code, these are discrete objects (integers or strings),
  usually represented with lists or enums.
- **Transitions:** Store as a dictionary mapping $(s, a)$ pairs to possible next
  states and their probabilities.
- **Rewards:** Also as a dict for specific $(s, a, s')$ transitions.
- **Policies:** Can be random (uniform), fixed (always one action), or
  probabilistic (describe action choice by state).

By implementing an MDP in Python, you can simulate any RL task with known ground
truth and test various policies systematically.

---

## Python Demonstrations

### Demo 1: Define a Small MDP as Python Data Structures

We'll define a tiny 3-state MDP with 2 actions.

```python
from typing import Dict, Tuple, List
import numpy as np

states: List[str] = ['A', 'B', 'C']
actions: List[str] = ['left', 'right']

# Transition probabilities: P[s][a] = list of (prob, next_state)
P: Dict[str, Dict[str, List[Tuple[float, str]]]] = {
    'A': {'left':  [(1.0, 'A')],
          'right': [(1.0, 'B')]},
    'B': {'left':  [(0.8, 'A'), (0.2, 'C')],
          'right': [(1.0, 'C')]},
    'C': {'left':  [(1.0, 'B')],
          'right': [(1.0, 'C')]}
}

# Rewards: R[s][a][s'] = reward
R: Dict[str, Dict[str, Dict[str, float]]] = {
    'A': {'left':  {'A': 0.0},
          'right': {'B': 1.0}},
    'B': {'left':  {'A': 0.0, 'C': 2.0},
          'right': {'C': 3.0}},
    'C': {'left':  {'B': 0.0},
          'right': {'C': 0.0}}
}
gamma: float = 0.9
```

---

### Demo 2: Simulate a Random Policy and Track Rewards

Let's run episodes under a random policy (choose actions uniformly in each
state).

```python
import random

def sample_next_state(s: str, a: str) -> str:
    p_list = P[s][a]
    probs, next_states = zip(*[(p, s2) for p, s2 in p_list])
    return random.choices(next_states, weights=probs)[0]

def get_reward(s: str, a: str, s2: str) -> float:
    return R[s][a][s2]

def run_episode(policy: Dict[str, str | None] = None, max_steps: int = 10) -> float:
    s = 'A'
    total_reward = 0.0
    trajectory = []
    for t in range(max_steps):
        a = policy[s] if policy and policy[s] else random.choice(actions)
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        total_reward += r
        trajectory.append((s, a, r, s2))
        s = s2
    print("Trajectory:", trajectory)
    print("Total reward:", total_reward)
    return total_reward

ep_rewards = [run_episode() for _ in range(5)]
print("Average reward:", np.mean(ep_rewards))
```

---

### Demo 3: Visualize State Transitions as a Graph

We'll use `networkx` to show how each (state, action) leads to other states.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.MultiDiGraph()
for s in P:
    for a in P[s]:
        for prob, s2 in P[s][a]:
            label = f"{a} ({prob})"
            G.add_edge(s, s2, label=label)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5,4))
nx.draw(G, pos, with_labels=True, node_size=1200, node_color="skyblue", font_weight="bold")
edge_labels = {(u,v,k['label']):k['label'] for u,v,k in G.edges(data=True)}
edge_labels = {(u,v): d['label'] for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("MDP State Transition Graph")
plt.show()
```

---

### Demo 4: Implement a Simple Policy and Compute Expected Reward

Let's try a greedy policy: always take `"right"`.

```python
# Define a deterministic policy
simple_policy = {s: "right" for s in states}

# Run several episodes and compute average reward
rewards_right = [run_episode(simple_policy) for _ in range(20)]
print("Right-only policy: avg reward =", np.mean(rewards_right))

# Try left-only for contrast
left_policy = {s: "left" for s in states}
rewards_left = [run_episode(left_policy) for _ in range(20)]
print("Left-only policy: avg reward =", np.mean(rewards_left))
```

---

## Exercises

### **Exercise 1:** Define a Small MDP as Python Data Structures

- Create a finite set of states (e.g., three rooms), two actions, transition
  probabilities, and rewards using dictionaries.

---

### **Exercise 2:** Simulate a Random Policy and Track Rewards

- Run several episodes with random actions.
- Print the trajectory (states, actions, rewards) and the total reward per
  episode.

---

### **Exercise 3:** Visualize State Transitions as a Graph

- Use `networkx` (or matplotlib) to draw the states and arrows for transitions,
  labelling by action and probability.

---

### **Exercise 4:** Implement a Simple Policy and Compute Expected Reward

- Devise a fixed policy (e.g., always go right).
- Simulate several episodes and calculate the mean total reward.
- Compare with another simple policy.

---

### **Sample Starter Code for Exercises**

```python
import random
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Exercise 1
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

# Exercise 2
def run_episode(policy=None, max_steps=8):
    s = 'A'; total = 0
    for t in range(max_steps):
        a = policy[s] if policy else random.choice(actions)
        s2 = sample_next_state(s,a)
        r = get_reward(s,a,s2)
        print(f't={t}, s={s}, a={a}, r={r}, s\'={s2}')
        total += r
        s = s2
    print('Total reward:', total)
run_episode()

# Exercise 3
G = nx.MultiDiGraph()
for s in P:
    for a in P[s]:
        for prob,s2 in P[s][a]:
            lbl = f"{a} ({prob})"
            G.add_edge(s, s2, label=lbl)
pos = nx.spring_layout(G, seed=2)
plt.figure(figsize=(5,4))
nx.draw(G, pos, with_labels=True, node_size=1300, node_color='orange')
edge_labels = {(u,v):d['label'] for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("State Transitions"); plt.show()

# Exercise 4
right_policy = {s: 'right' for s in states}
left_policy = {s: 'left' for s in states}
rewards_right = [run_episode(right_policy) for _ in range(10)]
rewards_left = [run_episode(left_policy) for _ in range(10)]
print("Avg reward right:", np.mean(rewards_right))
print("Avg reward left:", np.mean(rewards_left))
```

---

## Conclusion

You now know how to represent the key elements of RL—the MDP—in code and
visually! You can simulate random and fixed-action policies, trace how rewards
accumulate, and see the literal map of your agent’s world. Next, you'll tackle
_bandit problems_ and step further into the heart of RL: balancing exploration
and exploitation.

See you in Part 4.3!
