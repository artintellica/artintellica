from typing import Dict, Tuple, List
import torch

# State, action, next_state -> reward
TransitionDict = Dict[Tuple[int, int, int], float]

num_states: int = 3
num_actions: int = 2
gamma: float = 0.9

# Transition probabilities: P[s, a, s']
P: torch.Tensor = torch.zeros(
    (num_states, num_actions, num_states), dtype=torch.float32
)
# Reward function: R[s, a, s']
R: torch.Tensor = torch.zeros(
    (num_states, num_actions, num_states), dtype=torch.float32
)

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
policy: torch.Tensor = (
    torch.ones((num_states, num_actions), dtype=torch.float32) / num_actions
)

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

import matplotlib.pyplot as plt
import numpy as np

def plot_values_and_policy(
    values: torch.Tensor,
    policy: torch.Tensor,
    state_labels: List[str] | None = None,
    action_labels: List[str] | None = None
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
        ax.text(xs[s]+dx, 0.15, action_labels[best_a], ha="center", color="k", fontsize=16) # type: ignore

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, num_states-0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(state_labels)
    ax.axis("off")
    plt.title("State Values and Policy Actions")
    plt.show()

# Plot for greedy policy and its values
plot_values_and_policy(v_greedy, greedy_policy)
