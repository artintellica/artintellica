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
n_iterations = 100


# Function to run batch gradient descent with a given learning rate
def batch_gradient_descent(X, y, w_init, learning_rate, n_iterations):
    w = w_init.copy()
    losses = []
    for _ in range(n_iterations):
        # Compute gradient: (2/n) * X^T * (Xw - y)
        gradient = (2 / n_samples) * X.T @ (X @ w - y)
        # Update weights
        w = w - learning_rate * gradient
        # Compute and store loss (MSE)
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)
    return w, losses


# Run batch gradient descent with different learning rates
learning_rates = [0.001, 0.01, 0.1]
all_losses = []
final_weights = []

for lr in learning_rates:
    print(f"\nRunning Batch Gradient Descent with learning rate = {lr}")
    w_final, losses = batch_gradient_descent(X_with_bias, y, w_init, lr, n_iterations)
    all_losses.append(losses)
    final_weights.append(w_final)
    print(f"Final weights (bias, slope): {w_final}")

# Plot loss curves for each learning rate
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(range(n_iterations), all_losses[i], label=f"Learning Rate = {lr}")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error Loss")
plt.title("Loss Over Iterations for Different Learning Rates (Batch GD)")
plt.legend()
plt.grid(True)
plt.show()
