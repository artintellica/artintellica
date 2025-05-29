import numpy as np
import matplotlib.pyplot as plt

# Define the scaling matrix S
S = np.array([[2, 0], [0, 3]])

# Define the vector
vector = np.array([1, 1])

# Apply the scaling transformation
scaled_vector = S @ vector  # Matrix-vector multiplication

# Print results
print("Scaling matrix S:\n", S)
print("Original vector:", vector)
print("Scaled vector:", scaled_vector)


# Visualize original and scaled vectors
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Original and Scaled Vectors")
    plt.show()


plot_2d_vectors([vector, scaled_vector], ["Original", "Scaled"], ["blue", "red"])
