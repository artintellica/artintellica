import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a small dataset with 10 samples and 2 features
X = np.random.randn(10, 2)
print("Dataset (10 samples, 2 features):")
print(X)
print("Shape:", X.shape)

# Function to compute polynomial kernel matrix (degree=2)
def polynomial_kernel(X1, X2, degree=2):
    # Compute the inner product (dot product) between all pairs of points
    inner_product = np.dot(X1, X2.T)
    # Apply the polynomial kernel formula: k(x_i, x_j) = (1 + x_i^T x_j)^degree
    return (1 + inner_product) ** degree

# Compute the Gram matrix using the polynomial kernel
K = polynomial_kernel(X, X, degree=2)
print("\nPolynomial Kernel (Gram) Matrix (Degree=2, 10x10):")
print("Shape:", K.shape)
print(K)
