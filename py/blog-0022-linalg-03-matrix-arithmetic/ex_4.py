import numpy as np

# Create a 4x2 matrix
A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Create a 2D vector
v = np.array([1, -1])

# Add the vector to each row of the matrix using broadcasting
result = A + v

# Print results
print("Matrix A (4x2):\n", A)
print("\nVector v (2D):", v)
print("\nResult (A + v, broadcasted):\n", result)
