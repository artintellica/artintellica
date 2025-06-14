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
