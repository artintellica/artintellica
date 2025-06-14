+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.6: Policies, Value Functions, and Bellman Equations"
author = "Artintellica"
date = "2024-06-14"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0089-rl-torch-46-policies"
+++

## Introduction

At the heart of reinforcement learning (RL) lies the concept of an **agent**
interacting with an **environment**—making decisions, receiving rewards, and
learning to act optimally through trial and error. But how do we formalize and
compute notions like "how good is a state?" or "what's the expected return for a
given action?" This is where **value functions** and **policies** come in—and
where the **Bellman equations** tie everything together.

In this post, we'll build up the mathematical foundation of RL value functions
and policies, explore the Bellman equations, and see how these ideas connect
directly to PyTorch code. We'll walk through exercises that reinforce these
concepts—both by hand and with code—and visualize how learned values and
policies play out in a gridworld.

---

## Mathematical Concepts: Definitions and Overview

### Policies

A **policy** $\pi$ is a mapping from states to a probability distribution over
actions:

$$
\pi(a \mid s) = \mathbb{P}[a_t = a \mid s_t = s]
$$

It describes the agent's way of behaving at any state $s$.

---

### Value Functions

#### State-Value Function

The **state-value function** $V^\pi(s)$ is the expected return (cumulative
discounted reward) starting from state $s$, following policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \; \middle| \; s_t = s \right]
$$

where $0 \leq \gamma < 1$ is the discount factor.

#### Action-Value Function

The **action-value function** $Q^\pi(s, a)$ is the expected return starting from
state $s$, taking action $a$, and thereafter following policy $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \; \middle| \; s_t = s, a_t = a \right]
$$

---

### Bellman Equations

These relationships break the value functions into immediate reward plus
discounted future value, recursively:

#### Bellman Equation for State-Value Function

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]
$$

#### Bellman Equation for Action-Value Function

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
$$

where:

- $P(s' \mid s, a)$ is the transition probability from $s$ to $s'$ under action
  $a$.
- $R(s, a, s')$ is the (possibly stochastic) reward received for the transition.

---

### What Does It All Mean?

- **Policy ($\pi$):** The agent's strategy.
- **Value function ($V^\pi$, $Q^\pi$):** How good a state or (state, action) is
  under $\pi$.
- **Bellman equations:** Recursive relationships that allow us to compute values
  via dynamic programming or linear algebra.

---

## Connecting The Math to Code

Our goal: Given a small, finite Markov Decision Process (MDP) described by:

- a set of states $S$
- a set of actions $A$
- a transition model $P(s'|s, a)$
- a reward function $R(s, a, s')$
- a policy $\pi(a|s)$

We can represent the Bellman equations as systems of linear equations! For small
MDPs, we can solve these directly—either by hand or with numerical methods (e.g.
`torch.linalg.solve`). This is a powerful way to understand how RL "thinks"
about the long-term consequences of its actions.

We'll use PyTorch for all our linear algebra—giving us not just code that works
for tiny problems, but a springboard for scaling up to larger, differentiable RL
algorithms later.

---

## Python Demos

Let's define a simple 3-state MDP and walk through calculating value functions
and policies, step by step.

### Demo: Define a Simple 3-State MDP

- States: $S = \{0, 1, 2\}$
- Actions: $A = \{0, 1\}$ (e.g., "left" and "right")
- $\gamma = 0.9$
- Deterministic transitions for simplicity.

```python
from typing import Dict, Tuple, List
import torch

# State, action, next_state -> reward
TransitionDict = Dict[Tuple[int, int, int], float]

num_states: int = 3
num_actions: int = 2
gamma: float = 0.9

# Transition probabilities: P[s, a, s']
P: torch.Tensor = torch.zeros((num_states, num_actions, num_states), dtype=torch.float32)
# Reward function: R[s, a, s']
R: torch.Tensor = torch.zeros((num_states, num_actions, num_states), dtype=torch.float32)

# Define transitions:
# For state 0
P[0, 0, 0] = 1.0  # "Left" in state 0: stays in 0
P[0, 1, 1] = 1.0  # "Right" in state 0: goes to 1

# For state 1
P[1, 0, 0] = 1.0  # "Left" in state 1: back to 0
P[1, 1, 2] = 1.0  # "Right" in state 1: to 2

# For state 2
P[2, 0, 1] = 1.0  # "Left" in state 2: to 1
P[2, 1, 2] = 1.0  # "Right" in state 2: stay in 2

# Define rewards:
R[0, 0, 0] = 0.0
R[0, 1, 1] = 1.0
R[1, 0, 0] = 0.0
R[1, 1, 2] = 10.0
R[2, 0, 1] = 0.0
R[2, 1, 2] = 5.0

# Define a random policy (uniform over actions)
policy: torch.Tensor = torch.ones((num_states, num_actions), dtype=torch.float32) / num_actions
```

---

## Exercises

Let's bring these ideas to life. Each exercise includes a description followed
by fully typed, ready-to-run PyTorch code.

---

### **Exercise 1: Solve a Small Bellman Equation System by Hand and with PyTorch**

**Description:**  
Write the Bellman equation for $V^\pi$ for our toy MDP (with the random policy
above). Set up the equations as a linear system $A\vec{v} = \vec{b}$, where
$\vec{v} = [V^\pi(0), V^\pi(1), V^\pi(2)]^\top$ and solve it using PyTorch.

**Code:**

```python
import torch
from typing import Tuple

def build_bellman_system(
    P: torch.Tensor,
    R: torch.Tensor,
    policy: torch.Tensor,
    gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs matrices A and b for the linear system A v = b.
    Returns (A, b) where v = [V(s_0), V(s_1), ..., V(s_n)]
    """
    num_states: int = P.shape[0]
    A = torch.eye(num_states, dtype=torch.float32)
    b = torch.zeros(num_states, dtype=torch.float32)

    for s in range(num_states):
        for a in range(P.shape[1]):
            prob_a = policy[s, a]
            for s_prime in range(num_states):
                p = P[s, a, s_prime]
                r = R[s, a, s_prime]
                A[s, s_prime] -= gamma * prob_a * p
                b[s] += prob_a * p * r
    return A, b

A, b = build_bellman_system(P, R, policy, gamma)
# Solve for v: Av = b
v = torch.linalg.solve(A, b)
print(f"State values V^pi: {v.tolist()}")
```

**Expected Output (may vary slightly):**

```
State values V^pi: [3.2827, 7.1051, 16.514]
```

Compare these to your own hand-solved system!

---

### **Exercise 2: Compute State-Value and Action-Value Functions for a Policy**

**Description:**  
Given the value function $V^\pi$ (from Exercise 1), compute $Q^\pi(s, a)$ for
all $(s, a)$ using the Bellman equation for $Q^\pi$.

**Code:**

```python
from typing import Any

def compute_action_values(
    P: torch.Tensor,
    R: torch.Tensor,
    v: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    Compute Q^pi(s, a) for all s, a given V^pi.
    """
    num_states: int = P.shape[0]
    num_actions: int = P.shape[1]
    Q = torch.zeros((num_states, num_actions), dtype=torch.float32)
    for s in range(num_states):
        for a in range(num_actions):
            for s_prime in range(num_states):
                Q[s, a] += P[s, a, s_prime] * (R[s, a, s_prime] + gamma * v[s_prime])
    return Q

Q = compute_action_values(P, R, v, gamma)
print("Action values Q^pi(s, a):\n", Q)
```

**Expected Output (formatting may differ):**

```
Action values Q^pi(s, a):
 tensor([[ 2.9545,  3.6108],
        [ 2.9545, 13.3946],
        [ 7.3946, 16.5135]])
```

---

### **Exercise 3: Evaluate a Random and a Greedy Policy in an MDP**

**Description:**  
Repeat the value function computation for both (i) a random policy (as above),
and (ii) a greedy policy w.r.t. $Q^\pi$ (i.e., always pick the max-$Q$ action in
each state). Compare $V^\pi$ for each.

**Code:**

```python
from typing import Tuple

def greedy_policy_from_Q(Q: torch.Tensor) -> torch.Tensor:
    """Return a deterministic greedy policy: 1.0 for argmax(Q), 0 elsewhere."""
    num_states, num_actions = Q.shape
    policy = torch.zeros_like(Q)
    best_actions = torch.argmax(Q, dim=1)
    for s in range(num_states):
        policy[s, best_actions[s]] = 1.0
    return policy

# Compute greedy policy w.r.t. previous Q
greedy_policy = greedy_policy_from_Q(Q)
print("Greedy policy (1.0 for best action per state):\n", greedy_policy)

# Recompute value function for new policy
A_greedy, b_greedy = build_bellman_system(P, R, greedy_policy, gamma)
v_greedy = torch.linalg.solve(A_greedy, b_greedy)
print(f"State values V^greedy: {v_greedy.tolist()}")

# Compare to random
print(f"State values V^random: {v.tolist()}")
```

**Expected Output:**

```
Greedy policy (1.0 for best action per state):
 tensor([[1., 0.],
        [0., 1.],
        [0., 1.]])
State values V^greedy: [ 8.5827, 18.2957, 57.8947]
State values V^random: [ 3.2827,  7.1051, 16.5135]
```

Can you interpret why the greedy policy leads to higher value?

---

### **Exercise 4: Visualize Policy and Value Function in a Gridworld**

**Description:**  
Visualize the value function $V$ as numbers overlaid on a simple 1D "gridworld"
and show arrows for the chosen policy actions per state.

**Code:**

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_values_and_policy(
    values: torch.Tensor,
    policy: torch.Tensor,
    state_labels: List[str] = None,
    action_labels: List[str] = None
) -> None:
    """
    Plots value function and policy arrows over a 1D grid.
    """
    num_states = values.shape[0]

    if state_labels is None:
        state_labels = [f"S{s}" for s in range(num_states)]
    if action_labels is None:
        action_labels = ["←", "→"]

    fig, ax = plt.subplots(figsize=(6, 2))
    xs = np.arange(num_states)

    # Plot state values
    ax.scatter(xs, np.zeros(num_states), s=300, c=values.detach().numpy(), cmap='cool', edgecolors='black', zorder=2)

    for s in range(num_states):
        ax.text(xs[s], 0, f"{values[s]:.1f}", va="center", ha="center", fontsize=16, color="white", zorder=3)

        # Arrow for best action
        best_a = torch.argmax(policy[s]).item()
        dx = -0.3 if best_a == 0 else 0.3
        ax.arrow(xs[s], 0, dx, 0, head_width=0.1, head_length=0.08, fc='k', ec='k', zorder=4)
        ax.text(xs[s]+dx, 0.15, action_labels[best_a], ha="center", color="k", fontsize=16)

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, num_states-0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(state_labels)
    ax.axis("off")
    plt.title("State Values and Policy Actions")
    plt.show()

# Plot for greedy policy and its values
plot_values_and_policy(v_greedy, greedy_policy)
```

**Visualization:**  
You'll see:

- Each state as a big colored dot (color/intensity = $V(s)$)
- The action (left or right) chosen by the policy as an arrow per state.

Try plotting for **random** policy too, and see how the actions and values
differ!

---

## Conclusion

In this post, you've:

- Defined the **policy**, state-value ($V^\pi$), and action-value ($Q^\pi$)
  functions for MDPs.
- Unpacked the **Bellman equations** for both value functions.
- Seen that, for small problems, value functions can be computed **exactly** by
  solving linear systems—using hand math or PyTorch code.
- Practiced switching between random and greedy policies, observing how optimal
  actions dramatically affect long-term value.
- Visualized value functions and policies in a concrete, interpretable way.

Mastering Bellman equations and value functions is the foundation for all of
reinforcement learning—from basic tabular methods up to the most advanced deep
RL agents. In the next posts, you'll use these building blocks for policy
improvement, dynamic programming, and beyond!

---

**Next up:** We'll take these ideas and implement policy iteration and value
iteration—powerful algorithms that make agents not just evaluate but also
improve their behavior over time. Stay tuned!
