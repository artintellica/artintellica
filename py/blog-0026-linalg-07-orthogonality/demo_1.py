import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
u = np.array([1, 0])
v = np.array([0, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
print("Orthogonal?", np.isclose(dot_product, 0, atol=1e-10))


# Visualize vectors
def plot_2d_vectors(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


plot_2d_vectors([u, v], ["u", "v"], ["blue", "red"], "Orthogonal Vectors")
