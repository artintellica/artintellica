import numpy as np

# Create an array from a list
X = np.array([[1, 2, 3], [4, 5, 6]])

# Check the shape of an array
print("Shape of X:", X.shape)  # (2, 3)
print("Number of dimensions of X:", X.ndim)  # 2 (a 2D array/matrix)

# Element-wise addition
X_plus_5 = X + 5
print("X + 5:\n", X_plus_5)

# Element-wise multiplication
X_times_2 = X * 2
print("X * 2:\n", X_times_2)
