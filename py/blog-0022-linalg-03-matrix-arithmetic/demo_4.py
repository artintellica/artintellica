import numpy as np

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])

# Define a vector
v = np.array([1, -1])

# Add vector to each row of A
A_plus_v = A + v  # Broadcasting automatically expands v to match A's shape

# Print result
print("Vector v:", v)
print("Matrix A:\n", A)
print("A + v (broadcasted):\n", A_plus_v)
