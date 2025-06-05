import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load the digits dataset (1797 samples, 64 features)
digits = load_digits()
X, y = digits.data, digits.target
print("Original Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train and evaluate k-NN classifier on original data (64 dimensions)
start_time_original = time.time()
model_original = KNeighborsClassifier(n_neighbors=5)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
end_time_original = time.time()
runtime_original = end_time_original - start_time_original
print("\nResults on Original Data (64 dimensions):")
print(f"Accuracy: {accuracy_original:.4f}")
print(f"Runtime (seconds): {runtime_original:.4f}")

# Apply random projection to reduce to 10 dimensions
d = X.shape[1]  # Original dimension (64)
k = 10  # Target reduced dimension
R = np.random.randn(d, k) / np.sqrt(k)  # Gaussian random projection matrix
print("\nRandom Projection Matrix Shape:", R.shape)

# Project training and testing data
X_train_proj = X_train @ R
X_test_proj = X_test @ R
print("Projected Training Data Shape:", X_train_proj.shape)
print("Projected Testing Data Shape:", X_test_proj.shape)

# Train and evaluate k-NN classifier on projected data (10 dimensions)
start_time_projected = time.time()
model_projected = KNeighborsClassifier(n_neighbors=5)
model_projected.fit(X_train_proj, y_train)
y_pred_projected = model_projected.predict(X_test_proj)
accuracy_projected = accuracy_score(y_test, y_pred_projected)
end_time_projected = time.time()
runtime_projected = end_time_projected - start_time_projected
print("\nResults on Projected Data (10 dimensions):")
print(f"Accuracy: {accuracy_projected:.4f}")
print(f"Runtime (seconds): {runtime_projected:.4f}")

# Compare results
print("\nComparison:")
print(
    f"Original Data (64 dimensions) - Accuracy: {accuracy_original:.4f}, Runtime: {runtime_original:.4f} s"
)
print(
    f"Projected Data (10 dimensions) - Accuracy: {accuracy_projected:.4f}, Runtime: {runtime_projected:.4f} s"
)
print(
    f"Accuracy Difference (Original - Projected): {(accuracy_original - accuracy_projected):.4f}"
)
print(
    f"Runtime Difference (Original - Projected): {(runtime_original - runtime_projected):.4f} s"
)
