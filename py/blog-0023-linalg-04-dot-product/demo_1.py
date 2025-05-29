import numpy as np
import matplotlib.pyplot as plt

# Define two 2D vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)


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


plot_2d_vectors([u, v], ["u", "v"], ["blue", "red"])
