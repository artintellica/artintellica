from typing import Tuple, Dict, List
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
