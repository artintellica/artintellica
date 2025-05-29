import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 5x2 matrix representing 5 2D points
points = np.random.randint(low=-5, high=6, size=(5, 2))

# Define a query point
query = np.array([0, 0])

# Compute L2 (Euclidean) distances from query to each point
distances = np.linalg.norm(points - query, axis=1)

# Find the index of the closest point
closest_idx = np.argmin(distances)
closest_point = points[closest_idx]
closest_distance = distances[closest_idx]

# Print results
print("Points (5x2 matrix):\n", points)
print("\nQuery point:", query)
print("\nL2 distances:", distances)
print("\nClosest point:", closest_point)
print("Closest distance:", closest_distance)
print("Index of closest point:", closest_idx)
