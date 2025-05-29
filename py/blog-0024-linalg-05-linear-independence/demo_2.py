import numpy as np
import matplotlib.pyplot as plt

# Define two sets of 2D vectors
independent_vectors = np.array([[1, 0], [0, 1]])  # Linearly independent
dependent_vectors = np.array([[1, 2], [2, 4]])  # Linearly dependent

# Visualize span of independent vectors
def plot_2d_vectors(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    # Plot span as a shaded region (for independent vectors)
    if len(vectors) == 2 and np.linalg.det(vectors) != 0:
        # t = np.linspace(-10, 10, 100)
        for c1 in np.linspace(-2, 2, 20):
            for c2 in np.linspace(-2, 2, 20):
                point = c1 * vectors[0] + c2 * vectors[1]
                plt.scatter(point[0], point[1], color='gray', alpha=0.1, s=1)

    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

# Plot independent vectors
plot_2d_vectors(
    independent_vectors,
    ['v1', 'v2'],
    ['blue', 'red'],
    "Span of Linearly Independent Vectors"
)

# Plot dependent vectors (span is a line)
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, dependent_vectors[0, 0], dependent_vectors[0, 1], color='blue', scale=1, scale_units='xy', angles='xy')
plt.text(dependent_vectors[0, 0], dependent_vectors[0, 1], 'v1, v2', color='blue', fontsize=12)
t = np.linspace(-3, 3, 100)
line = t[:, np.newaxis] * dependent_vectors[0]
plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.5)
plt.grid(True)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Span of Linearly Dependent Vectors (Line)")
plt.show()
