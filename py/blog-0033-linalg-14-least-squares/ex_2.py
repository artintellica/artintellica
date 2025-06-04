import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(45)

# Generate the same 2D dataset (1 feature, 30 samples)
n_samples = 30
X = np.random.randn(n_samples, 1) * 1.5  # Feature
y = 2.5 * X[:, 0] + 1.0 + np.random.randn(n_samples) * 0.3  # Target with noise
print("Dataset shape:", X.shape, y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])


# Function to compute the mean squared error loss
def compute_loss(X, y, w):
    predictions = X @ w
    return np.mean((predictions - y) ** 2)


# Gradient Descent implementation
def gradient_descent(X, y, w_init, learning_rate, n_iterations):
    w = w_init.copy()
    losses = []
    for _ in range(n_iterations):
        # Compute gradient: 2 * X^T * (Xw - y) / n_samples
        gradient = 2 * X.T @ (X @ w - y) / len(y)
        # Update weights
        w = w - learning_rate * gradient
        # Record loss
        loss = compute_loss(X, y, w)
        losses.append(loss)
    return w, losses


# Initialize weights to zeros
w_init = np.zeros(2)
n_iterations = 100

# Experiment with different learning rates
learning_rates = [0.001, 0.01, 0.1]
all_losses = []
final_weights = []

for lr in learning_rates:
    print(f"\nRunning Gradient Descent with learning rate = {lr}")
    w_final, losses = gradient_descent(X_with_bias, y, w_init, lr, n_iterations)
    all_losses.append(losses)
    final_weights.append(w_final)
    print(f"Final weights (bias, slope): {w_final}")

# Plot loss over iterations for each learning rate
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(range(n_iterations), all_losses[i], label=f"Learning Rate = {lr}")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error Loss")
plt.title("Loss Over Iterations for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()
