import numpy as np

# Set random seed for reproducibility
np.random.seed(50)

# Generate a synthetic 4D dataset (3 features + bias, 100 samples)
n_samples = 100
# Generate three features
X1 = np.random.randn(n_samples) * 2.0  # First feature
X2 = np.random.randn(n_samples) * 1.5  # Second feature
X3 = np.random.randn(n_samples) * 1.0  # Third feature
# Target is a linear combination of all features with noise
true_weights = [2.5, 1.0, -1.5, 0.5]  # [bias, coef_X1, coef_X2, coef_X3]
y = (
    true_weights[0]
    + true_weights[1] * X1
    + true_weights[2] * X2
    + true_weights[3] * X3
    + np.random.randn(n_samples) * 0.3
)
# Combine features into a matrix
X = np.vstack([X1, X2, X3]).T
print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
print("Shape of X with bias term:", X_with_bias.shape)

# Initialize weights
w_init = np.zeros(4)  # 4 weights: bias + 3 features
eta = 0.01  # Learning rate
n_iterations = 1000  # More iterations for SGD to converge

# Stochastic Gradient Descent (SGD)
w = w_init.copy()
for _ in range(n_iterations):
    # Randomly select one sample
    idx = np.random.randint(0, n_samples)
    X_sample = X_with_bias[idx : idx + 1]  # Shape (1, 4)
    y_sample = y[idx : idx + 1]  # Shape (1,)
    # Compute gradient for single sample: 2 * X^T * (Xw - y)
    gradient = 2 * X_sample.T @ (X_sample @ w - y_sample)
    # Update weights
    w = w - eta * gradient

# Print final weights compared to true weights
print("\nFinal weights (bias, coef_X1, coef_X2, coef_X3):", w)
print("True weights (bias, coef_X1, coef_X2, coef_X3):", true_weights)
