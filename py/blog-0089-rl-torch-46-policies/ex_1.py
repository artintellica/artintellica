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
