import numpy as np

# Create an array from a list
X = np.array([[1, 2, 3], [4, 5, 6]])
print("Array from list:\n", X)

# Create an array of zeros (useful for initializing biases or placeholders)
zeros = np.zeros((2, 3))
print("Array of zeros:\n", zeros)

# Create an array of random numbers (useful for initializing weights)
W = np.random.randn(3, 2)  # Standard normal distribution (mean=0, std=1)
print("Random array:\n", W)
