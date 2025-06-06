import numpy as np

# Input matrix X of shape (4, 2) - 4 samples, 2 features each
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
print("Input matrix X (4x2):\n", X)

# Weight matrix W of shape (2, 3) - mapping 2 input features to 3 output features
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])
print("Weight matrix W (2x3):\n", W)

# Compute Z = X @ W, resulting in shape (4, 3)
Z = X @ W
print("Output matrix Z (4x3):\n", Z)
