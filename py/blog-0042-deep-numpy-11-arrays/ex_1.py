import numpy as np

ones = np.ones((4, 3))
print("Original array of ones:\n", ones)
print("Shape of ones:", ones.shape)  # (4, 3)
reshaped = ones.reshape(3, 4)
print("Reshaped array to 3x4:\n", reshaped)
print("Shape of reshaped ones:", reshaped.shape)  # (3, 4)
