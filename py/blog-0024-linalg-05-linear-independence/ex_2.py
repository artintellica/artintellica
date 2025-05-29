import numpy as np
import matplotlib.pyplot as plt

# Create two 2D vectors (example: linearly independent)
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Check linear independence using determinant of the matrix [v1, v2]
vectors = np.array([v1, v2]).T  # 2x2 matrix
det = np.linalg.det(vectors)

# Print vectors and independence status
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Determinant of [v1, v2]:", det)
print("Linearly independent?", not np.isclose(det, 0, atol=1e-10))


# Visualize span
def plot_2d_span(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)

    # Plot vectors
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    # Check if linearly independent (non-zero determinant)
    if not np.isclose(np.linalg.det(np.array(vectors).T), 0, atol=1e-10):
        # Plot span as a plane (for independent vectors)
        t = np.linspace(-10, 10, 100)
        for c1 in np.linspace(-2, 2, 20):
            for c2 in np.linspace(-2, 2, 20):
                point = c1 * vectors[0] + c2 * vectors[1]
                plt.scatter(point[0], point[1], color="gray", alpha=0.1, s=1)
    else:
        # Plot span as a line (for dependent vectors)
        t = np.linspace(-3, 3, 100)
        line = t[:, np.newaxis] * vectors[0]  # Use first vector (non-zero)
        plt.plot(line[:, 0], line[:, 1], "gray", alpha=0.5)

    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


# Plot span
plot_2d_span([v1, v2], ["v1", "v2"], ["blue", "red"], "Span of Two 2D Vectors")

# Example with dependent vectors
v1_dep = np.array([1, 2])
v2_dep = np.array([2, 4])  # v2 = 2 * v1
plot_2d_span(
    [v1_dep, v2_dep],
    ["v1", "v2"],
    ["blue", "red"],
    "Span of Linearly Dependent Vectors",
)
