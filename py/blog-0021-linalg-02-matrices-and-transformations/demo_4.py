import numpy as np
import matplotlib.pyplot as plt
import torch

# Define a 2D vector
vector = np.array([1, 0])

# Define a 90-degree rotation matrix (pi/2 radians)
theta = np.pi / 2
rotation_matrix = np.array(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
)

# Apply transformation
rotated_vector = rotation_matrix @ vector  # Matrix-vector multiplication

# Print results
print("Original vector:", vector)
print("Rotation matrix:\n", rotation_matrix)
print("Rotated vector:", rotated_vector)


# Visualize original and rotated vectors
def plot_2d_vectors(vectors, labels, colors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units="xy", angles="xy")
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.title("Vector Rotation")
    plt.show()


plot_2d_vectors([vector, rotated_vector], ["Original", "Rotated"], ["blue", "red"])

# Convert to PyTorch tensors
vector_torch = torch.tensor([1.0, 0.0])
rotation_matrix_torch = torch.tensor(
    [
        [torch.cos(torch.tensor(np.pi / 2)), -torch.sin(torch.tensor(np.pi / 2))],
        [torch.sin(torch.tensor(np.pi / 2)), torch.cos(torch.tensor(np.pi / 2))],
    ]
)

# Matrix-vector multiplication
rotated_vector_torch = rotation_matrix_torch @ vector_torch

print("PyTorch rotated vector:", rotated_vector_torch)
