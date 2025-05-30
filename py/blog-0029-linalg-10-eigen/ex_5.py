import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create a 5x2 matrix of 5 2D points
points = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 4]])

# Center the points by subtracting the mean
mean = np.mean(points, axis=0)
points_centered = points - mean

# Compute the covariance matrix
cov_matrix = np.cov(points_centered.T, bias=True)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Print results
print("Points (5x2):\n", points)
print("\nMean of points:", mean)
print("\nCentered points:\n", points_centered)
print("\nCovariance matrix:\n", cov_matrix)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)

# Visualize points and principal directions
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], c="blue", label="Points")
plt.scatter(mean[0], mean[1], c="red", s=100, label="Mean")

# Plot principal directions scaled by sqrt(eigenvalues) for visibility
scale = 2  # Scaling factor for visualization
for i in range(2):
    eig_vec = eigenvectors[:, i]
    eig_val = np.sqrt(eigenvalues[i])
    plt.quiver(
        mean[0],
        mean[1],
        eig_vec[0] * eig_val * scale,
        eig_vec[1] * eig_val * scale,
        color=["green", "purple"][i],
        scale=1,
        scale_units="xy",
        angles="xy",
    )
    plt.text(
        mean[0] + eig_vec[0] * eig_val * scale,
        mean[1] + eig_vec[1] * eig_val * scale,
        f"e{i+1}",
        color=["green", "purple"][i],
        fontsize=12,
    )

plt.grid(True)
plt.xlim(-2, 7)
plt.ylim(-1, 7)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Points and Principal Directions")
plt.legend()
plt.show()
