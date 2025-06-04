import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(47)

# Generate a small dataset (10 samples, 1 feature)
n_samples = 10
X = np.random.randn(n_samples, 1) * 2  # Feature
# True relationship is quadratic with noise
y = 0.5 * X[:, 0] ** 2 + 1.5 * X[:, 0] + 1.0 + np.random.randn(n_samples) * 0.5
print("Dataset shape:", X.shape, y.shape)

# Initialize lists to store MSE for each polynomial degree
degrees = range(1, 6)  # Degrees 1 to 5
train_mse = []

# Fit models for each polynomial degree
for degree in degrees:
    # Transform features to polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict on training data
    y_pred = model.predict(X_poly)

    # Compute and store MSE for training data
    mse = mean_squared_error(y, y_pred)
    train_mse.append(mse)
    print(f"Degree {degree} - Training MSE: {mse:.4f}")

# Plot training MSE for each degree
plt.figure(figsize=(8, 6))
plt.plot(degrees, train_mse, marker="o", label="Training MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training MSE vs. Polynomial Degree")
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()
