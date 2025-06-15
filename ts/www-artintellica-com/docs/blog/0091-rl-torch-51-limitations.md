+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.1: Limitations of Tabular RL and the Need for Function Approximation"
author = "Artintellica"
date = "2024-06-14"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0091-rl-torch-51-limitations"
+++

# Introduction

Welcome back to **Learn Reinforcement Learning with PyTorch**! Up to now, we've
explored key reinforcement learning concepts—states, actions, rewards—and
implemented classic “tabular” algorithms such as Q-Learning and SARSA. These
approaches store the value estimates for _every possible_ state (or state-action
pair) in a table. This method works well for simple environments, but quickly
becomes infeasible as problems scale.

In this post, we’ll uncover the **limitations of tabular reinforcement
learning** (RL), especially the “curse of dimensionality”, and explain why
function approximation (especially deep neural networks) is critical for modern
RL. You’ll learn the mathematics behind the state explosion, see hands-on demos,
and begin thinking about designing neural architectures to replace (or
generalize) the idea of the Q-table.

# Mathematical Overview

## The Curse of Dimensionality in Tabular RL

In tabular RL, we represent the action-value function $Q(s, a)$ (or state-value
$V(s)$) as a table, explicitly storing an entry for _each_ state (or
state-action) pair.

Suppose an environment’s state is described by $n$ discrete variables, with each
variable $i$ ranging over $k_i$ possible values. Then, the **total number of
possible states** is

$$
N_{\text{states}} = \prod_{i=1}^n k_i
$$

For each action $a$ from a set $\mathcal{A}$, we track $Q(s, a)$:

$$
N_{\text{table entries}} = N_{\text{states}} \times |\mathcal{A}|
$$

This grows **exponentially** with the number of state variables (“dimensions”).

### Example

If an agent observes a $5 \times 5$ grid (think of a small Gridworld), with its
location as 2 variables x and y (each with 5 possible values), then:

$$
N_{\text{states}} = 5 \times 5 = 25
$$

—no problem! But if you increase to an $N\times N$ grid (large map, $N=100$), or
the agent observes more features (positions of several objects),
$N_{\text{states}}$ explodes.

If each variable has 100 possible values, and there are 4 variables:

$$
N_{\text{states}} = 100^4 = 100,000,000
$$

Modern RL environments (e.g., Atari games, robotics with images) can have
_billions_ or more possible states!

## Why Does Tabular RL Fail in Large State Spaces?

- **Memory:** We cannot store tables with billions of entries.
- **Learning:** Most entries never get visited—wasting data, slow learning.
- **Generalization:** Each state is treated as unrelated; the agent can’t
  generalize experiences to similar states.

## Function Approximation to the Rescue

Rather than represent $Q(s, a)$ as a lookup table, we learn a _function_:

$$
Q(s, a) \approx f_\theta(s, a)
$$

where $f_\theta$ is a parameterized model (e.g., a neural network with weights
$\theta$), mapping input state $s$ and action $a$ to an estimated value $Q$.
This enables:

- **Compactness:** Model size does not grow exponentially with state dimension.
- **Generalization:** Agent can interpolate/extrapolate to unseen states.
- **Scalability:** The same approach can work for images, high-dimensional
  robotics, etc.

# From Math to Code: Building Intuition

Let’s see what state and Q-table explosion look like in code, and why neural
networks are needed.

- For each dimension added to a discrete state space, the table size
  _multiplies_.
- The table is just a big PyTorch or NumPy array—which quickly becomes gigantic.
- Neural networks can output $Q(s,a)$ for any $(s,a)$ pair without storing all
  values explicitly.

# Demo 1: Creating an Impractically Large Q-table

Let's illustrate how quickly table size explodes as the environment becomes more
complex.

```python
from typing import List, Tuple
import torch

def compute_table_size(state_sizes: List[int], n_actions: int) -> int:
    """
    Compute total entries for a Q-table given state variable sizes and number of actions.
    """
    from functools import reduce
    from operator import mul
    n_states = reduce(mul, state_sizes, 1)
    return n_states * n_actions

# Example: 4 state variables, each with 100 values; 5 actions
state_sizes: List[int] = [100, 100, 100, 100]
n_actions: int = 5
table_entries: int = compute_table_size(state_sizes, n_actions)
print(f"Total Q-table entries: {table_entries:,}")
```

**Output:**

```
Total Q-table entries: 500,000,000
```

Allocating a table this size—with, say, 32-bit floats—would require ~2GB just
for Q-values! Imagine 10 variables; it’s not possible.

# Demo 2: Exponential State Growth Visualization

Let’s visualize how the number of states grows as we increase the number of
variables (“dimensions”).

```python
import matplotlib.pyplot as plt

def plot_state_space_growth(k: int, max_vars: int) -> None:
    """
    Plot number of possible states vs number of state variables for discrete variables.
    """
    import numpy as np
    num_vars = list(range(1, max_vars+1))
    num_states = [k ** n for n in num_vars]
    plt.figure(figsize=(7,4))
    plt.semilogy(num_vars, num_states, marker='o')
    plt.title(f"Exponential Growth of State Space (k={k})")
    plt.xlabel("Number of state variables (dimensions)")
    plt.ylabel("Number of possible states (log scale)")
    plt.grid(True, which='both')
    plt.show()

plot_state_space_growth(k=10, max_vars=10)
```

Try it! As variables go up, state space grows _exponentially_ (don't forget to
use log-scale axes!).

# Demo 3: Attempt Tabular Q-Learning in a Large State Space

Let’s try running tabular Q-learning in a simple “large” toy environment—and see
what happens.

```python
from typing import Tuple, Dict
import random
import numpy as np

# Our toy environment: 6 discrete state variables, each with 10 possible values
state_sizes: List[int] = [10] * 6  # 10^6 = 1,000,000 states
n_actions: int = 4

# Simulate the environment as a tuple of 6 ints
def random_state() -> Tuple[int, ...]:
    return tuple(random.randint(0, 9) for _ in range(6))

# Initialize a Q-table
Q_table: Dict[Tuple[int,...], np.ndarray] = {}

# Try updating Q-table for 100,000 steps
for step in range(100_000):
    state = random_state()
    action = random.randint(0, n_actions-1)
    next_state = random_state()
    reward = random.random()
    # Q-update (SARS, fixed alpha/gamma for this demo)
    q_old = Q_table.get(state, np.zeros(n_actions))
    alpha = 0.1
    gamma = 0.99
    # Q-learning update
    q_target = reward + gamma * np.max(Q_table.get(next_state, np.zeros(n_actions)))
    q_new = q_old.copy()
    q_new[action] = (1 - alpha) * q_old[action] + alpha * q_target
    Q_table[state] = q_new

print(f"Q-table size after 100,000 transitions: {len(Q_table):,} states visited")
print(f"Fraction of possible states visited: {len(Q_table)/1_000_000:.2%}")
```

**What do you think will happen?** We’ll find that after 100,000 episodes, only
about 10% of the **state space** was ever even visited—most of the state-action
values were never updated at all!

# Demo 4: Designing a Neural Network to Replace the Q-table

Suppose we want $Q(s,a)$ for arbitrary states $s$ (which could be vectors,
images, etc). Here's a minimal PyTorch MLP architecture that maps state (and
action) to $Q(s, a)$. For discrete actions, we often have our network output a
vector of Q-values, one per action.

```python
import torch
import torch.nn as nn
from typing import Any

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Input: state tensor of shape (batch_size, state_dim)
        Output: Q-values, tensor of shape (batch_size, n_actions)
        """
        return self.net(state)

# Example: state has 6 features, 4 possible actions
state_dim: int = 6
n_actions: int = 4
q_net = QNetwork(state_dim, n_actions)
# Batch of states
states: torch.Tensor = torch.rand(8, state_dim)
q_values: torch.Tensor = q_net(states)
print(q_values.shape)  # (8, 4)
```

This neural net’s parameters don’t grow with the number of possible states—they
are fixed! The learned function **generalizes** to new, even unseen, states.

# Exercises

---

## Exercise 1: Create a Large State Space Where Q-tables are Impractical

**Task:**  
Write code to define an environment where there are 8 discrete state variables,
each with 20 possible values. Compute how many total states there are, and how
large (in MB) a Q-table would be if you stored a 32-bit float for each $(s,a)$
pair for 6 discrete actions.

**Solution:**

```python
from typing import List
import numpy as np

def q_table_size(state_sizes: List[int], n_actions: int, dtype: np.dtype = np.float32) -> float:
    from functools import reduce
    from operator import mul
    n_states = reduce(mul, state_sizes, 1)
    total_entries = n_states * n_actions
    bytes_total = total_entries * np.dtype(dtype).itemsize
    mb_total = bytes_total / (1024**2)
    return n_states, total_entries, mb_total

state_sizes: List[int] = [20] * 8  # 20^8
n_actions: int = 6

n_states, n_entries, size_mb = q_table_size(state_sizes, n_actions)
print(f"Number of states: {n_states:,}")
print(f"Q-table entries: {n_entries:,}")
print(f"Q-table size: {size_mb:,.2f} MB (float32)")
```

---

## Exercise 2: Visualize Exponential State Growth in a Simple Environment

**Task:**  
For a sequence of 1 to 12 state variables, each having 10 possible values, plot
(on a log scale) the number of possible states versus the number of variables.
Annotate the plot to show where the state count exceeds 1 million.

**Solution:**

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_state_growth(base: int = 10, max_vars: int = 12) -> None:
    num_vars = np.arange(1, max_vars+1)
    num_states = base ** num_vars
    plt.figure(figsize=(7,4))
    plt.semilogy(num_vars, num_states, marker='o')
    plt.axhline(1_000_000, color='red', linestyle='--', label="1 million states")
    plt.title(f"Exponential State Space Growth (base={base})")
    plt.xlabel("Number of state variables")
    plt.ylabel("Number of possible states (log scale)")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

plot_state_growth()
```

---

## Exercise 3: Attempt Tabular Learning and Analyze Why It Fails

**Task:**  
Simulate tabular Q-learning in an environment with 6 discrete state variables
(each with 10 values), updating the Q-table for 250,000 random transitions.
Afterward, report:

- How many unique states were visited out of possible $10^6$.
- What fraction of the table is still unused?

**Solution:**

```python
from typing import Tuple, Dict, List
import random
import numpy as np

state_sizes: List[int] = [10] * 6  # 10^6 possible states
n_actions: int = 4
total_states: int = 10 ** 6

def random_state() -> Tuple[int, ...]:
    return tuple(random.randint(0, 9) for _ in range(6))

Q_table: Dict[Tuple[int,...], np.ndarray] = {}

for _ in range(250_000):
    state = random_state()
    action = random.randint(0, n_actions-1)
    next_state = random_state()
    reward = random.random()
    q_old = Q_table.get(state, np.zeros(n_actions))
    alpha = 0.1
    gamma = 0.99
    q_target = reward + gamma * np.max(Q_table.get(next_state, np.zeros(n_actions)))
    q_new = q_old.copy()
    q_new[action] = (1 - alpha) * q_old[action] + alpha * q_target
    Q_table[state] = q_new

visited = len(Q_table)
unused = total_states - visited
print(f"Unique states visited: {visited:,} / {total_states:,} ({visited/total_states:.2%})")
print(f"Fraction of table remaining unused: {unused/total_states:.2%}")
```

---

## Exercise 4: Propose a Neural Network Architecture to Replace Q-table

**Task:**  
Specify (in PyTorch code) a neural network that takes an input state vector of
size 8 and outputs Q-values for 6 actions. Use at least two hidden layers. Print
the number of parameters.

**Solution:**

```python
import torch
import torch.nn as nn

class BigQNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, n_actions: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

qnet = BigQNetwork()
n_params = sum(p.numel() for p in qnet.parameters())
print(f"Q-network has {n_params:,} parameters (trainable)")
# Example usage: for a batch of 10 states
state_batch = torch.rand(10, 8)
q_outputs = qnet(state_batch)
print(f"Output shape: {q_outputs.shape}")  # (10, 6)
```

---

# Conclusion

Tabular RL is a great tool for learning, but quickly reaches its limits in
practical problems, falling victim to the **curse of dimensionality**. The
exponential growth in the number of states makes storing (and learning) a
separate value for each state-action pair impossible.

**Function approximation**—using neural networks—is how modern RL methods scale
to large, complex environments. Neural nets can generalize from observed to
unobserved states, store Q or value functions compactly, and enable deep RL to
work with high-dimensional states like images and physics sensors.

**Up next:** We’ll dive into implementing _Deep Q-Networks (DQN)_ in PyTorch,
your first step towards high-performance, scalable RL agents!
