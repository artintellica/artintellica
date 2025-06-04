import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(49)

# Generate a new 2D dataset (1 feature, 40 samples)
n_samples = 40
X = np.random.randn(n_samples, 1) * 1.8  # Feature
y = 2.0 * X[:, 0] + 1.5 + np.random.randn(n_samples) * 0.4  # Target with noise
print("Dataset shape:", X.shape, y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])

# Initialize weights
w_init = np.zeros(2)
eta = 0.01  # Learning rate
n_iterations = 100

# Batch Gradient Descent
w = w_init.copy()
losses = []
for _ in range(n_iterations):
    # Compute gradient: (2/n) * X^T * (Xw - y)
    gradient = (2 / n_samples) * X_with_bias.T @ (X_with_bias @ w - y)
    # Update weights
    w = w - eta * gradient
    # Compute and store loss (MSE)
    loss = np.mean((X_with_bias @ w - y) ** 2)
    losses.append(loss)

print("Final weights (bias, slope):", w)
print("True weights (bias, slope): [1.5, 2.0]")

# Plot loss over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(n_iterations), losses, label="Loss (MSE)")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Loss Over Iterations (Batch Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()
