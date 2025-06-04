import numpy as np

# Set random seed for reproducibility
np.random.seed(46)

# Generate a synthetic 3D dataset (2 features + bias, 50 samples)
n_samples = 50
# Generate two features
X1 = np.random.randn(n_samples) * 2  # First feature
X2 = np.random.randn(n_samples) * 1.5  # Second feature
# Target is a linear combination of both features with noise
y = 1.5 * X1 - 2.0 * X2 + 3.0 + np.random.randn(n_samples) * 0.5
# Combine features into a matrix
X = np.vstack([X1, X2]).T
print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
print("Shape of X with bias term:", X_with_bias.shape)

# Solve using normal equations: w = (X^T X)^(-1) X^T y
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
w = np.linalg.solve(XTX, XTy)  # More stable than direct inverse
print("\nLearned weights (bias, coefficient for X1, coefficient for X2):", w)
print("True weights (bias, coefficient for X1, coefficient for X2): [3.0, 1.5, -2.0]")
