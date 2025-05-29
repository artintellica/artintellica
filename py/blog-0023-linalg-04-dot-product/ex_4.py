import numpy as np
import matplotlib.pyplot as plt

# Create two 2D orthogonal vectors (dot product = 0)
u = np.array([1, 0])
v = np.array([0, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Compute cosine similarity
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
cosine_sim = dot_product / (norm_u * norm_v) if norm_u * norm_v != 0 else 0

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
print("Cosine similarity:", cosine_sim)


# Visualize vectors in 2D
def plot_2d_vectors(vectors, labels, colors):
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
    plt.title("Orthogonal Vectors")
    plt.show()


plot_2d_vectors([u, v], ["u", "v"], ["blue", "red"])
