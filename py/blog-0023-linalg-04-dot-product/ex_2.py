import numpy as np
import matplotlib.pyplot as plt

# Vectors from Exercise 1 (using same random seed for consistency)
np.random.seed(42)
u = np.random.randint(low=-5, high=6, size=3)
v = np.random.randint(low=-5, high=6, size=3)

# Compute dot product
dot_product = np.dot(u, v)

# Compute norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

# Compute cosine similarity
cosine_sim = dot_product / (norm_u * norm_v) if norm_u * norm_v != 0 else 0

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
print("Norm of u:", norm_u)
print("Norm of v:", norm_v)
print("Cosine similarity:", cosine_sim)


# Visualize first two components in 2D
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(
            *origin, *vec[:2], color=color, scale=1, scale_units="xy", angles="xy"
        )
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vectors in 2D (First Two Components)")
    plt.show()


plot_2d_vectors([u, v], ["u", "v"], ["blue", "red"])
