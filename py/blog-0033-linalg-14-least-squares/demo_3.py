import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate a simple 2D dataset (1 feature + bias)
np.random.seed(42)
n_samples = 50
X = np.random.randn(n_samples, 1) * 2  # Feature
y = 3 * X[:, 0] + 2 + np.random.randn(n_samples) * 0.5  # Target with noise
print("Dataset shape:", X.shape, y.shape)


# Use scikit-learn's LinearRegression
model = LinearRegression()
model.fit(X, y)
print("scikit-learn weights (bias, slope):", [model.intercept_, model.coef_[0]])

# Predict and plot
y_pred_sk = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label="Data Points")
plt.plot(X[:, 0], y_pred_sk, "b-", label="Fitted Line (scikit-learn)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with scikit-learn")
plt.legend()
plt.grid(True)
plt.show()
