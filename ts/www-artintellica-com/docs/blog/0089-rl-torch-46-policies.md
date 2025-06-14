+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.6: Policies, Value Functions, and Bellman Equations"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

You've explored value estimation via Monte Carlo and TD, but what are we really estimating? At the heart of RL are **value functions** and the **Bellman equations**, which build the mathematical backbone for all value-based RL. Understanding and *solving* Bellman equations helps demystify why value learning works at all.

In this post, you’ll:

- Solve a small Bellman system by hand and with PyTorch/Numpy
- Compute both state-value ($V$) and action-value ($Q$) functions for a given policy
- Evaluate random and greedy policies in simple MDPs
- Visualize policies and value functions in grid worlds

---

## Mathematics: Value Functions and Bellman Equations

### **State-Value Function**
For a policy $\pi$, the value function is:
\[
V^\pi(s) = \mathbb{E}_\pi \left[\, \sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s \,\right]
\]

### **Bellman Equation for $V^\pi$ (for a finite MDP):**
\[
V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [\, R(s, a, s') + \gamma V^\pi(s') \,]
\]
This is a *system of equations* (one for each state).

### **Action-Value Function**
\[
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) [\, R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \,]
\]

### **Bellman Optimality Equation**
For optimal policy $\pi^*$:
\[
V^*(s) = \max_a \sum_{s'} P(s'|s, a)\left[ R(s, a, s') + \gamma V^*(s') \right]
\]

---

## Explanation: How the Math Connects to Code

- To **solve** Bellman’s equations, encode them as a system of linear equations and solve for $V$ using Numpy/PyTorch.
- **Policy evaluation:** Plug a known $\pi(a|s)$ into the formula to compute $V^\pi$ and $Q^\pi$.
- **Policy comparison:** Use $V^\pi$ to compare total expected rewards for different policies.
- In gridworlds, you can plot $V$ as a heatmap, and policies as arrows.

---

## Python Demonstrations

### Demo 1: Solve a Small Bellman Equation System by Hand and with PyTorch

Suppose an MDP:
- States: $A$, $B$
- Actions: "stay" (0), "go" (1)
- Policy: Always pick "go" ($\pi(1|s)=1$)
- Transition: From $A$, "go" $\to$ $B$ (reward=1); from $B$, "go" $\to$ $A$ (reward=2)
- $\gamma=0.9$

\[
\begin{align*}
V(A) &= R(A,1,B) + \gamma V(B) \\
V(B) &= R(B,1,A) + \gamma V(A)
\end{align*}
\]
where $R(A,1,B)=1$, $R(B,1,A)=2$.

This gives:

\[
\begin{align*}
V(A) &= 1 + 0.9 V(B) \\
V(B) &= 2 + 0.9 V(A)
\end{align*}
\]

#### Hand solution:
Substitute one into the other,
\[
V(A) = 1 + 0.9 [2 + 0.9 V(A)] = 1 + 1.8 + 0.81 V(A) \\
V(A) - 0.81 V(A) = 2.8 \\
0.19 V(A) = 2.8 \implies V(A) \approx 14.74 \\
V(B) = 2 + 0.9 \cdot 14.74 \approx 15.27
\]

#### PyTorch solution:

```python
import torch

gamma = 0.9
# [V(A), V(B)]
# V(A) = 1 + gamma * V(B)
# V(B) = 2 + gamma * V(A)
A = torch.tensor([[1.0, -gamma], [-gamma, 1.0]])
b = torch.tensor([1.0, 2.0])
V = torch.linalg.solve(A, b)
print("V(A) = %.2f, V(B) = %.2f" % (V[0], V[1]))
```

---

### Demo 2: Compute State-Value and Action-Value Functions for a Policy

Expand to 3-state MDP (as in earlier posts):

```python
import numpy as np

states = ['A', 'B', 'C']
nS = len(states)
actions = ['left', 'right']
nA = len(actions)

# Policy: always "right"
pi = {s: {'left': 0.0, 'right': 1.0} for s in states}

# Transition and reward as before
P = {
    'A': {'left':  [(1.0, 'A')], 'right': [(1.0, 'B')]},
    'B': {'left':  [(0.8, 'A'), (0.2, 'C')], 'right': [(1.0, 'C')]},
    'C': {'left':  [(1.0, 'B')], 'right': [(1.0, 'C')]}
}
R = {
    'A': {'left':  {'A': 0.0}, 'right': {'B': 1.0}},
    'B': {'left':  {'A': 0.0, 'C': 2.0}, 'right': {'C': 3.0}},
    'C': {'left':  {'B': 0.0}, 'right': {'C': 0.0}}
}
gamma = 0.9

# Build Bellman equations for V_pi (policy evaluation)
A = np.eye(nS)
b = np.zeros(nS)
for i, s in enumerate(states):
    for a in actions:
        pa = pi[s][a]
        for prob, s2 in P[s][a]:
            j = states.index(s2)
            r = R[s][a][s2]
            A[i, j] -= gamma * pa * prob
            b[i] += pa * prob * r

V = np.linalg.solve(A, b)
print("V under given policy:", dict(zip(states, V)))

# Q-function for all state-action pairs
Q = np.zeros((nS, nA))
for i, s in enumerate(states):
    for k, a in enumerate(actions):
        for prob, s2 in P[s][a]:
            j = states.index(s2)
            r = R[s][a][s2]
            # expected next state's V
            Q[i, k] += prob * (r + gamma * V[j])

print("Q-table:\n", Q)
```

---

### Demo 3: Evaluate a Random and a Greedy Policy in an MDP

```python
# Policy 1: always right
policy_right = {s: 'right' for s in states}
# Policy 2: always left
policy_left = {s: 'left' for s in states}

def evaluate_policy(policy, n_episodes=50):
    tot = 0.0
    for _ in range(n_episodes):
        s = 'A'
        for t in range(5):
            a = policy[s]
            s2 = np.random.choice([spp for _, spp in P[s][a]], p=[p for p, _ in P[s][a]])
            r = R[s][a][s2]
            tot += r
            s = s2
    return tot / n_episodes

print("Avg reward (always right):", evaluate_policy(policy_right))
print("Avg reward (always left):", evaluate_policy(policy_left))
# Random
def random_policy(s): return np.random.choice(actions)
def evaluate_random_policy(n_episodes=50):
    tot = 0.0
    for _ in range(n_episodes):
        s = 'A'
        for t in range(5):
            a = random_policy(s)
            s2 = np.random.choice([spp for _, spp in P[s][a]], p=[p for p, _ in P[s][a]])
            r = R[s][a][s2]
            tot += r
            s = s2
    return tot / n_episodes
print("Avg reward (random policy):", evaluate_random_policy())
```

---

### Demo 4: Visualize Policy and Value Function in a Gridworld

Make a simple gridworld, solve for $V$, and plot as a heatmap, arrows for policy.

```python
# 2x2 Gridworld example
import matplotlib.pyplot as plt
import numpy as np

size = 2
states = [(i,j) for i in range(size) for j in range(size)]
state_idx = {s: i for i,s in enumerate(states)}
nS = len(states)
actions = ['up', 'down', 'left', 'right']
a_to_delta = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
gamma = 0.8

# Terminal at (1,1) with reward=5
def step(s,a):
    if s==(1,1): return s, 0, True
    di,dj = a_to_delta[a]
    s2 = (min(size-1,max(0,s[0]+di)), min(size-1,max(0,s[1]+dj)))
    r = 5.0 if s2==(1,1) else -0.1
    done = s2==(1,1)
    return s2, r, done

# Random policy
pi = {s: 'right' if s[1]<size-1 else 'down' for s in states}

# Value Iteration to solve for V*
V = np.zeros(nS)
for _ in range(50):
    V_new = np.zeros_like(V)
    for si, s in enumerate(states):
        vals = []
        for a in actions:
            s2, r, _ = step(s,a)
            V2 = V[state_idx[s2]]
            vals.append(r + gamma*V2)
        V_new[si] = max(vals)
    V = V_new

V_grid = V.reshape(size, size)
plt.imshow(V_grid, cmap='viridis')
plt.colorbar(label="V(s)")
plt.title("Optimal Value Function (2x2 Grid)")
plt.show()

# Visualize policy
plt.imshow(V_grid, cmap='viridis')
for si, s in enumerate(states):
    a = pi[s]
    i,j = s
    di,dj = a_to_delta[a]
    plt.arrow(j, i, dj*0.3, di*0.3, head_width=0.08, color='red')
plt.title("Policy on Gridworld")
plt.gca().invert_yaxis()
plt.show()
```

---

## Exercises

### **Exercise 1:** Solve a Small Bellman Equation System by Hand and with PyTorch/Numpy

- Set up the Bellman equations for a 2-state, 1-action-per-state MDP.
- Solve algebraically and with `torch.linalg.solve` or `np.linalg.solve`.

---

### **Exercise 2:** Compute State-Value and Action-Value Functions for a Policy

- For a small MDP, encode the transitions and a policy.
- Compute $V^\pi$ and $Q^\pi$ for all states/actions, using Bellman equations.

---

### **Exercise 3:** Evaluate a Random and a Greedy Policy in an MDP

- Simulate episodes for each policy.
- Compute and compare average rewards.

---

### **Exercise 4:** Visualize Policy and Value Function in a Gridworld

- Use matplotlib to plot state values on a grid, and show the greedy action in each cell as an arrow.

---

### **Sample Starter Code**

```python
import numpy as np
import torch

# EXERCISE 1
A = torch.tensor([[1.0, -0.8], [-0.8, 1.0]])
b = torch.tensor([1.0, 2.0])
V = torch.linalg.solve(A, b)
print("Solution for V:", V.tolist())

# EXERCISE 2
# See Demo 2 above

# EXERCISE 3
# See Demo 3 above

# EXERCISE 4
# See Demo 4 above
```

---

## Conclusion

You now see how the **Bellman equations** unify value learning, why they underlie RL algorithms, and how to solve them for a given policy or the optimal one. This forms the essential toolkit for value-based planning and RL.

Next: We’ll move into hands-on RL agent engineering—designing, running, and diagnosing your own Gridworld agent!

See you in Part 4.7!
