import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D dataset with two non-linearly separable classes (moon shapes, same as Exercise 3)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

print("Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Define parameter grid for C and gamma (for RBF kernel)
param_grid = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.01, 0.1, 1.0, 10.0]}

# Create SVM classifier with RBF kernel
svm_rbf = SVC(kernel="rbf", random_state=42)

# Perform grid search with cross-validation (5-fold)
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X, y)

# Print the best parameters and the best accuracy score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Compute accuracy on the training set using the best model
train_accuracy = best_model.score(X, y)
print("Training Accuracy (Best Model):", train_accuracy)


# Function to plot decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


# Plot decision boundary for the best model
plot_decision_boundary(
    X,
    y,
    best_model,
    f'SVM with RBF Kernel (Best Model: C={grid_search.best_params_["C"]}, gamma={grid_search.best_params_["gamma"]})',
)
