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

