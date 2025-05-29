import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create two 3D vectors with random integers between -5 and 5
u = np.random.randint(low=-5, high=6, size=3)
v = np.random.randint(low=-5, high=6, size=3)

# Compute dot product
dot_product = np.dot(u, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
