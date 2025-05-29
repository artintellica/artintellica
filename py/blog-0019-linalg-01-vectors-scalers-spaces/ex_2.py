import numpy as np
import matplotlib.pyplot as plt

# Define the vectors
a = np.array([3, -2])
b = np.array([-1, 4])

# Compute sum and difference
vector_sum = a + b
vector_diff = a - b

# Print results
print("Vector a:", a)
print("Vector b:", b)
print("a + b =", vector_sum)
print("a - b =", vector_diff)


# Visualize vectors in 2D
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)  # Origin point [0, 0]

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


# Plot vectors a, b, their sum, and difference
plot_2d_vectors(
    [a, b, vector_sum, vector_diff],
    ["a", "b", "a+b", "a-b"],
    ["blue", "red", "green", "purple"],
)
