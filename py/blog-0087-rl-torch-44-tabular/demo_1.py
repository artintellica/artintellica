import numpy as np

n_states = 4
n_actions = 2
Q = np.zeros((n_states, n_actions))  # Q[state, action]

print("Initial Q-table:")
print(Q)
