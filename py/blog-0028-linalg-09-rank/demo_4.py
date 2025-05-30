import numpy as np
import matplotlib.pyplot as plt


# Define a 2x3 matrix
A_vis = np.array([[1, 0, 1], [0, 1, 1]])

# Compute rank
rank_vis = np.linalg.matrix_rank(A_vis)

# Plot column vectors
plt.figure(figsize=(6, 6))
origin = np.zeros(2)
for i in range(A_vis.shape[1]):
    plt.quiver(
        *origin,
        *A_vis[:, i],
        color=["blue", "red", "green"][i],
        scale=1,
        scale_units="xy",
        angles="xy",
    )
    plt.text(A_vis[0, i], A_vis[1, i], f"col{i+1}", fontsize=12)

# If rank = 2, span is the plane
if rank_vis == 2:
    t = np.linspace(-2, 2, 20)
    for c1 in t:
        for c2 in t:
            point = c1 * A_vis[:, 0] + c2 * A_vis[:, 1]
            plt.scatter(*point, color="gray", alpha=0.1, s=1)

plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Column Space (Rank = {rank_vis})")
plt.show()
