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

# Gradient Descent for the same dataset
w_init = np.zeros(2)  # Initial weights (bias, slope)
eta = 0.01  # Learning rate
n_iterations = 100

w_gd = w_init.copy()
for _ in range(n_iterations):
    gradient = 2 * X_with_bias.T @ (X_with_bias @ w_gd - y) / n_samples
    w_gd = w_gd - eta * gradient

print("Learned weights with Gradient Descent (bias, slope):", w_gd)

# Predict and plot
y_pred_gd = X_with_bias @ w_gd
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label="Data Points")
plt.plot(X[:, 0], y_pred_gd, "g-", label="Fitted Line (Gradient Descent)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
