from typing import Tuple, Dict, List
import random
import numpy as np

state_sizes: List[int] = [10] * 6  # 10^6 possible states
n_actions: int = 4
total_states: int = 10**6


def random_state() -> Tuple[int, ...]:
    return tuple(random.randint(0, 9) for _ in range(6))


Q_table: Dict[Tuple[int, ...], np.ndarray] = {}

for _ in range(250_000):
    state = random_state()
    action = random.randint(0, n_actions - 1)
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
print(
    f"Unique states visited: {visited:,} / {total_states:,} ({visited/total_states:.2%})"
)
print(f"Fraction of table remaining unused: {unused/total_states:.2%}")
