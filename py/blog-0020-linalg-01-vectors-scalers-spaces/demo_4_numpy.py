import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
u = np.array([1, 2])
v = np.array([2, -1])

# Addition
sum_uv = u + v
print("u + v =", sum_uv)

# Scaling
scaled_v = 2 * v
print("2 * v =", scaled_v)


def plot_3d_vectors(vectors, labels, colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    origin = np.zeros(3)

    for vec, label, color in zip(vectors, labels, colors):
        ax.quiver(*origin, *vec, color=color)
        ax.text(vec[0], vec[1], vec[2], label, color=color)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Vector Visualization")
    plt.show()


# 3D vectors
u_3d = np.array([1, 2, 3])
v_3d = np.array([2, -1, 1])
plot_3d_vectors([u_3d, v_3d], ["u", "v"], ["blue", "red"])
