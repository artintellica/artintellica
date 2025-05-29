import numpy as np

# Define matrices A and B
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])

# Compute matrix products AB and BA
AB = A @ B  # Matrix multiplication
BA = B @ A

# Print results
print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("\nAB:\n", AB)
print("\nBA:\n", BA)

# Verify non-commutativity
is_commutative = np.array_equal(AB, BA)
print("\nIs AB equal to BA? (Is multiplication commutative?):", is_commutative)
