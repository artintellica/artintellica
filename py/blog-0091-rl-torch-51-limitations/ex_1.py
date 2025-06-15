from typing import List, Tuple
import numpy as np


def q_table_size(
    state_sizes: List[int], n_actions: int, dtype: str = "float32"
) -> Tuple[int, int, float]:
    """
    Computes (number of states, number of table entries, table size in MB).
    dtype: can be str like 'float32', or any valid np.dtype.
    """
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
