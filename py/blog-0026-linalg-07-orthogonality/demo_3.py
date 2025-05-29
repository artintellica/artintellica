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
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


# Define two linearly independent vectors
v1 = np.array([1, 1])
v2 = np.array([1, 0])

# Gram-Schmidt process
u1 = v1 / np.linalg.norm(v1)  # Normalize v1
w2 = v2 - np.dot(v2, u1) * u1  # Orthogonalize v2
u2 = w2 / np.linalg.norm(w2)  # Normalize w2

# Verify orthonormality
print("Orthonormal basis:")
print("u1:", u1)
print("u2:", u2)
print("u1 · u2:", np.dot(u1, u2))
print("Norm u1:", np.linalg.norm(u1))
print("Norm u2:", np.linalg.norm(u2))

# Visualize
plot_2d_vectors(
    [v1, v2, u1, u2],
    ["v1", "v2", "u1", "u2"],
    ["blue", "red", "green", "purple"],
    "Gram-Schmidt Orthonormal Basis",
)
