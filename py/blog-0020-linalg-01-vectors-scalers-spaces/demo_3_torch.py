import torch
import matplotlib.pyplot as plt

# Define two vectors
u = torch.tensor([1, 2], dtype=torch.float32)
v = torch.tensor([2, -1], dtype=torch.float32)

# Addition
sum_uv = u + v
print("u + v =", sum_uv)

# Scaling
scaled_v = 2 * v
print("2 * v =", scaled_v)


def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = torch.zeros(2)  # Origin point [0, 0]

    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.title("2D Vector Visualization")
    plt.show()


# Plot u, v, and their sum
plot_2d_vectors([u, v, sum_uv], ["u", "v", "u+v"], ["blue", "red", "green"])
