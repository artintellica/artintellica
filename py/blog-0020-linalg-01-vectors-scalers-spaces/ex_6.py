import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Create a 3D synthetic data point (height, weight, age)
vector = np.array([170, 70, 30])  # Example: height (cm), weight (kg), age (years)

# Normalize the vector by dividing by its maximum value
max_value = np.max(vector)
normalized_vector = vector / max_value

# Print original and normalized vectors
print("Original vector:", vector)
print("Normalized vector:", normalized_vector)
print("Maximum value used for normalization:", max_value)


# Plot original and normalized vectors in 3D
def plot_3d_vectors(vectors, labels, colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    origin = np.zeros(3)

    for vec, label, color in zip(vectors, labels, colors):
        ax.quiver(*origin, *vec, color=color)
        ax.text(vec[0], vec[1], vec[2], label, color=color)

    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 2])
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")
    ax.set_zlabel("Age")
    plt.title("3D Vector Visualization (Original and Normalized)")
    plt.show()


# Plot both vectors
plot_3d_vectors(
    [vector, normalized_vector], ["Original", "Normalized"], ["blue", "red"]
)
