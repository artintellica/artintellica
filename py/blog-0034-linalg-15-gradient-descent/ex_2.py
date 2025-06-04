import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(49)

# Generate the same 2D dataset (1 feature, 40 samples)
n_samples = 40
X = np.random.randn(n_samples, 1) * 1.8  # Feature
y = 2.0 * X[:, 0] + 1.5 + np.random.randn(n_samples) * 0.4  # Target with noise
print("Dataset shape:", X.shape, y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])

# Initialize weights
w_init = np.zeros(2)
eta = 0.01  # Learning rate
n_iterations_sgd = 500  # More iterations since updates are noisier

# Stochastic Gradient Descent (SGD)
w_sgd = w_init.copy()
losses_sgd = []
for _ in range(n_iterations_sgd):
    # Randomly select one sample
    idx = np.random.randint(0, n_samples)
    X_sample = X_with_bias[idx : idx + 1]  # Shape (1, 2)
    y_sample = y[idx : idx + 1]  # Shape (1,)
    # Compute gradient for single sample: 2 * X^T * (Xw - y)
    gradient = 2 * X_sample.T @ (X_sample @ w_sgd - y_sample)
    # Update weights
    w_sgd = w_sgd - eta * gradient
    # Compute and store loss on full dataset for monitoring
    loss = np.mean((X_with_bias @ w_sgd - y) ** 2)
    losses_sgd.append(loss)

print("Final weights with SGD (bias, slope):", w_sgd)
print("True weights (bias, slope): [1.5, 2.0]")

# Batch Gradient Descent (from Exercise 1, for comparison)
n_iterations_batch = 100
w_batch = w_init.copy()
losses_batch = []
for _ in range(n_iterations_batch):
    gradient = (2 / n_samples) * X_with_bias.T @ (X_with_bias @ w_batch - y)
    w_batch = w_batch - eta * gradient
    loss = np.mean((X_with_bias @ w_batch - y) ** 2)
    losses_batch.append(loss)

print("Final weights with Batch GD (bias, slope):", w_batch)

# Plot loss over iterations for both methods
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations_sgd), losses_sgd, label="SGD Loss (MSE)", alpha=0.5)
plt.plot(
    range(n_iterations_batch), losses_batch, label="Batch GD Loss (MSE)", linewidth=2
)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Loss Over Iterations: SGD vs Batch Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
