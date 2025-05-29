import numpy as np
import matplotlib.pyplot as plt

u = np.array([1, 2])


# Visualize vectors
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vectors for Dot Product")
    plt.show()


# Define vectors with varying similarity
v1 = np.array([1, 0])  # Same direction as u
v2 = np.array([0, 1])  # Orthogonal to v1
v3 = np.array([-1, 0])  # Opposite to v1

# Compute cosine similarities
cos_sim_v1 = np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1))
cos_sim_v2 = np.dot(u, v2) / (np.linalg.norm(u) * np.linalg.norm(v2))
cos_sim_v3 = np.dot(u, v3) / (np.linalg.norm(u) * np.linalg.norm(v3))

# Print results
print("Cosine similarity u, v1:", cos_sim_v1)
print("Cosine similarity u, v2:", cos_sim_v2)
print("Cosine similarity u, v3:", cos_sim_v3)

# Plot vectors
plot_2d_vectors(
    [u, v1, v2, v3],
    ["u", "v1 (similar)", "v2 (orthogonal)", "v3 (opposite)"],
    ["blue", "green", "red", "purple"],
)
