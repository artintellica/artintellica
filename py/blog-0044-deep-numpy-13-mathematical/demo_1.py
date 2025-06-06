import numpy as np

# Create a small matrix of values
Z = np.array([[0, 1, -1], [2, -2, 0.5]])
print("Input matrix Z (2x3):\n", Z)

# Apply exponential element-wise
exp_Z = np.exp(Z)
print("Exponential of Z (e^Z):\n", exp_Z)
