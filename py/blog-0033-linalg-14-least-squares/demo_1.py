import numpy as np
import matplotlib.pyplot as plt

# Generate a simple 2D dataset (1 feature + bias)
np.random.seed(42)
n_samples = 50
X = np.random.randn(n_samples, 1) * 2  # Feature
y = 3 * X[:, 0] + 2 + np.random.randn(n_samples) * 0.5  # Target with noise
print("Dataset shape:", X.shape, y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])

# Solve using normal equations: w = (X^T X)^(-1) X^T y
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
w = np.linalg.solve(XTX, XTy)  # More stable than direct inverse
print("Learned weights (bias, slope):", w)

# Predict and plot
y_pred = X_with_bias @ w
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label="Data Points")
plt.plot(X[:, 0], y_pred, "r-", label="Fitted Line (Normal Equations)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Normal Equations")
plt.legend()
plt.grid(True)
plt.show()
