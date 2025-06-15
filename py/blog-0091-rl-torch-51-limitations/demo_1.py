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
