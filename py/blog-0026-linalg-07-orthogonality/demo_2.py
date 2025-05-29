import numpy as np
import matplotlib.pyplot as plt


# Visualize vectors
def plot_2d_vectors(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


# Define vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute projection of u onto v
dot_uv = np.dot(u, v)
norm_v_squared = np.sum(v**2)
projection = (dot_uv / norm_v_squared) * v

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Projection of u onto v:", projection)

# Visualize
plot_2d_vectors(
    [u, v, projection],
    ["u", "v", "proj_v(u)"],
    ["blue", "red", "green"],
    "Projection of u onto v",
)
