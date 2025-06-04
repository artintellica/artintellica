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

# Initialize weights
w_init = np.zeros(2)
eta = 0.01  # Learning rate
n_iterations = 100

# Stochastic Gradient Descent for the same dataset
w = w_init.copy()
losses_sgd = []
n_iterations_sgd = 500  # More iterations since updates are noisier

for _ in range(n_iterations_sgd):
    # Randomly select one sample
    idx = np.random.randint(0, n_samples)
    X_sample = X_with_bias[idx:idx+1]  # Shape (1, 2)
    y_sample = y[idx:idx+1]  # Shape (1,)
    # Compute gradient for single sample: 2 * X^T * (Xw - y)
    gradient = 2 * X_sample.T @ (X_sample @ w - y_sample)
    # Update weights
    w = w - eta * gradient
    # Compute and store loss on full dataset for monitoring
    loss = np.mean((X_with_bias @ w - y) ** 2)
    losses_sgd.append(loss)

print("Final weights with SGD (bias, slope):", w)

# Plot loss over iterations
plt.figure(figsize=(8, 6))
plt.plot(range(n_iterations_sgd), losses_sgd, label='Loss (MSE)', alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Loss Over Iterations (Stochastic Gradient Descent)')
plt.legend()
plt.grid(True)
plt.show()
