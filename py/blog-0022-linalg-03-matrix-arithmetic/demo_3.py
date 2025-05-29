import numpy as np

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
AB = A @ B  # or np.matmul(A, B)

# Print result
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A @ B:\n", AB)
