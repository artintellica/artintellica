import numpy as np
import matplotlib.pyplot as plt

# Define the matrix
A = np.array([[4, 1], [1, 3]])

# Create a grid of x1, x2 values
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

# Compute the quadratic form x^T A x
for i in range(len(x1)):
    for j in range(len(x2)):
        x = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = x.T @ A @ x

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x^T A x')
ax.set_title('Quadratic Form for Positive Definite Matrix')
plt.show()
