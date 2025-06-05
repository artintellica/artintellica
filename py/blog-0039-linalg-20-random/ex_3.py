import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset (1000 samples, 200 features)
n_samples, n_features = 1000, 200
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
print("Original Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train and evaluate logistic regression on original data
model_original = LogisticRegression(random_state=42, max_iter=1000)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print("\nAccuracy on Original Data (200 dimensions):", accuracy_original)

# Apply random projection to reduce to 20 dimensions
k = 20  # Target reduced dimension
R = np.random.randn(n_features, k) / np.sqrt(k)  # Gaussian random projection matrix
print("Random Projection Matrix Shape:", R.shape)

# Project training and testing data
X_train_proj = X_train @ R
X_test_proj = X_test @ R
print("Projected Training Data Shape:", X_train_proj.shape)
print("Projected Testing Data Shape:", X_test_proj.shape)

# Train and evaluate logistic regression on projected data
model_projected = LogisticRegression(random_state=42, max_iter=1000)
model_projected.fit(X_train_proj, y_train)
y_pred_projected = model_projected.predict(X_test_proj)
accuracy_projected = accuracy_score(y_test, y_pred_projected)
print("Accuracy on Projected Data (20 dimensions):", accuracy_projected)

# Compare accuracies
print("\nAccuracy Comparison:")
print(f"Original Data (200 dimensions): {accuracy_original:.4f}")
print(f"Projected Data (20 dimensions): {accuracy_projected:.4f}")
print(
    f"Difference (Original - Projected): {(accuracy_original - accuracy_projected):.4f}"
)
