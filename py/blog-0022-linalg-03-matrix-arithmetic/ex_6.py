import numpy as np
import matplotlib.pyplot as plt

# Create a 3x2 matrix (each column is a 3D vector, but we'll plot in 2D)
A = np.array([[1, 2], [3, 4], [0, 0]])  # Third component is 0 for 2D plotting

# Create two 2D vectors (weights for linear combination)
v1 = np.array([1, 0])  # Weight for first column
v2 = np.array([0, 1])  # Weight for second column

# Compute linear combination: A @ v1 and A @ v2, and combine
result = A @ v1  # Linear combination with v1 (first column of A)
result2 = A @ v2  # Linear combination with v2 (second column of A)
combined_result = A @ (v1 + v2)  # Linear combination with v1 + v2

# Print results
print("Matrix A (3x2):\n", A)
print("\nVector v1:", v1)
print("\nVector v2:", v2)
print("\nLinear combination A @ v1:", result)
print("\nLinear combination A @ v2:", result2)
print("\nLinear combination A @ (v1 + v2):", combined_result)


# Visualize in 2D (using only first two components)
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(
            *origin, *vec[:2], color=color, scale=1, scale_units="xy", angles="xy"
        )
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Combinations of Matrix Columns")
    plt.show()


# Plot columns of A and the combined result
plot_2d_vectors(
    [A[:, 0], A[:, 1], combined_result],
    ["A_col1", "A_col2", "Result"],
    ["blue", "red", "green"],
)
