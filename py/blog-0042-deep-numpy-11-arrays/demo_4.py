import numpy as np

# Create an array from a list
X = np.array([[1, 2, 3], [4, 5, 6]])

# Broadcasting a scalar across an array
bias = 10
X_with_bias = X + bias
print("X with broadcasted bias:\n", X_with_bias)

# Broadcasting a 1D array across rows
row_bias = np.array([1, 2, 3])
X_with_row_bias = X + row_bias
print("X with row bias broadcasted:\n", X_with_row_bias)
