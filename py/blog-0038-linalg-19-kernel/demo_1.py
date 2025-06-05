import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D data with two non-linearly separable classes (moon shapes)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
print("Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Create and train an SVM with RBF kernel
svm_rbf = SVC(kernel="rbf", C=1.0, random_state=42)
svm_rbf.fit(X, y)


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


# Plot decision boundary for RBF kernel SVM
plot_decision_boundary(X, y, svm_rbf, "SVM with RBF Kernel Decision Boundary")

# Compute accuracy
accuracy = svm_rbf.score(X, y)
print("Training Accuracy (RBF Kernel SVM):", accuracy)
