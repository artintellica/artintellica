import numpy as np
import matplotlib.pyplot as plt

# Define vectors
u = np.array([1, 2])  # Original vector u from the blog post
v = np.array([2, -1])  # Original vector v from the blog post
sum_uv = u + v  # Sum of u and v from the blog post
scaled_u = 2 * u  # Scaled version of u (by scalar 2)


# Modified 2D plotting function
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)  # Origin point [0, 0]

    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    plt.grid(True)
    plt.xlim(-5, 5)  # Adjusted limits to accommodate scaled vector
    plt.ylim(-5, 5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.title("2D Vector Visualization with Scaled Vector")
    plt.show()


# Plot u, v, their sum, and scaled u
plot_2d_vectors(
    [u, v, sum_uv, scaled_u],
    ["u", "v", "u+v", "2u"],
    ["blue", "red", "green", "purple"],
)
