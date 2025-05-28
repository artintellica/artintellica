"""
exercise_4_graph_laplacian_diffusion.py
-------------------------------------------------
Construct the Laplacian for a 6-node ring graph.
Simulate diffusion of an initial delta on node 0 for 10 steps,
plot value on each node at each step.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 6  # number of nodes

# --- Build adjacency matrix for 6-node ring
A = np.zeros((N, N))
for i in range(N):
    A[i, (i - 1) % N] = 1  # left neighbor
    A[i, (i + 1) % N] = 1  # right neighbor

# --- Degree matrix
D = np.diag(A.sum(axis=1))

# --- Laplacian: L = D - A
L = D - A

# --- Diffusion simulation
steps = 10
D_coef = 0.5  # diffusion rate (should be < 1 for stability)
history = np.zeros((steps + 1, N))
u = np.zeros(N)
u[0] = 1.0  # initial delta at node 0
history[0] = u.copy()

for t in range(1, steps + 1):
    u = u - D_coef * (L @ u)
    history[t] = u

# --- Plot
plt.figure(figsize=(7, 4))
for node in range(N):
    plt.plot(range(steps + 1), history[:, node], marker="o", label=f"node {node}")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("Diffusion on 6-node Ring Graph (Graph Laplacian)")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: print Laplacian matrix
print("Graph Laplacian (6-node ring):\n", L)
