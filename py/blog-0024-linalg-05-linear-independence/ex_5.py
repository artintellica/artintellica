import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create two linearly independent 3D vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Generate random coefficients between -1 and 1
c1 = np.random.uniform(low=-1, high=1)
c2 = np.random.uniform(low=-1, high=1)

# Compute linear combination
result = c1 * v1 + c2 * v2

# Print results
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Coefficient c1:", c1)
print("Coefficient c2:", c2)
print("Linear combination result:", result)


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
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Combination (First Two Components)")
    plt.show()


plot_2d_vectors(
    [v1, v2, result], ["v1", "v2", "c1*v1 + c2*v2"], ["blue", "red", "green"]
)
