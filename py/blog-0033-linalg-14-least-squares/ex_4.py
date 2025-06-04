import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility (same as Exercise 3)
np.random.seed(46)

# Generate the same synthetic 3D dataset (2 features + bias, 50 samples)
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

# Normal Equations solution (from Exercise 3)
# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
# Solve using normal equations: w = (X^T X)^(-1) X^T y
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
w_normal = np.linalg.solve(XTX, XTy)
print(
    "\nNormal Equations weights (bias, coefficient for X1, coefficient for X2):",
    w_normal,
)

# Predict using normal equations solution
y_pred_normal = X_with_bias @ w_normal
# Compute MSE for normal equations
mse_normal = mean_squared_error(y, y_pred_normal)
print("Mean Squared Error (Normal Equations):", mse_normal)

# scikit-learn Linear Regression
model = LinearRegression()
model.fit(X, y)
# Extract weights (bias is intercept_, coefficients are coef_)
w_sklearn = [model.intercept_] + list(model.coef_)
print(
    "\nscikit-learn weights (bias, coefficient for X1, coefficient for X2):", w_sklearn
)

# Predict using scikit-learn model
y_pred_sklearn = model.predict(X)
# Compute MSE for scikit-learn
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
print("Mean Squared Error (scikit-learn):", mse_sklearn)

# Compare the results
print("\nComparison of weights:")
print(f"Normal Equations: {w_normal}")
print(f"scikit-learn:     {w_sklearn}")
print("\nComparison of MSE:")
print(f"Normal Equations MSE: {mse_normal:.6f}")
print(f"scikit-learn MSE:    {mse_sklearn:.6f}")
