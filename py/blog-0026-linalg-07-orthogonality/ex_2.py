import numpy as np
import matplotlib.pyplot as plt

# Define vectors
u = np.array([2, 3])  # Vector to project
v = np.array([1, 1])  # Vector to project onto

# Compute projection of u onto v
dot_uv = np.dot(u, v)  # Dot product u · v
norm_v_squared = np.sum(v**2)  # Squared length of v (v · v)
projection = (dot_uv / norm_v_squared) * v  # Projection formula

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_uv)
print("Squared length of v:", norm_v_squared)
print("Projection of u onto v:", projection)

# Visualize vectors and projection
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Projection of u onto v")
    plt.show()

plot_2d_vectors(
    [u, v, projection],
    ['u', 'v', 'proj_v(u)'],
    ['blue', 'red', 'green']
)
