import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D dataset with two linearly separable classes
n_samples = 200
# Class 1: points centered around (2, 2)
class1 = np.random.normal(loc=[2, 2], scale=1.0, size=(n_samples // 2, 2))
# Class 2: points centered around (-2, -2)
class2 = np.random.normal(loc=[-2, -2], scale=1.0, size=(n_samples // 2, 2))
# Combine the data
X = np.vstack([class1, class2])
# Labels: 0 for class1, 1 for class2
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

print("Data Shape:", X.shape)
print("Labels Shape:", y.shape)

# Create and train an SVM with linear kernel
svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_linear.fit(X, y)


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


# Plot decision boundary for Linear Kernel SVM
plot_decision_boundary(X, y, svm_linear, "SVM with Linear Kernel Decision Boundary")

# Compute accuracy
accuracy = svm_linear.score(X, y)
print("Training Accuracy (Linear Kernel SVM):", accuracy)
