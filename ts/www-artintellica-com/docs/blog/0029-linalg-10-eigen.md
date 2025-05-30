+++
title = "Linear Algebra for Machine Learning, Part 10: Eigenvalues and Eigenvectors"
author = "Artintellica"
date = "2025-05-30"
+++


# Linear Algebra for Machine Learning, Part 10: Eigenvalues and Eigenvectors

Welcome to the tenth post in our series on **Linear Algebra for Machine Learning**, continuing Part II: Core Theorems and Algorithms! After exploring rank, nullspace, and the Fundamental Theorem, we now dive into **eigenvalues** and **eigenvectors**, powerful concepts that underpin covariance analysis, principal component analysis (PCA), stability analysis, and spectral clustering in machine learning (ML). In this post, we’ll cover the mathematical foundations, their ML applications, and how to implement them in Python using **NumPy** and **PyTorch**. We’ll include visualizations to provide geometric intuition and Python exercises to deepen your understanding.

---

## The Math: Eigenvalues and Eigenvectors

### Definitions
For a square matrix \( A \) (size \( n \times n \)), a scalar \( \lambda \) is an **eigenvalue**, and a non-zero vector \( \mathbf{v} \) is an **eigenvector** if:

\[
A \mathbf{v} = \lambda \mathbf{v}
\]

This means applying \( A \) to \( \mathbf{v} \) scales \( \mathbf{v} \) by \( \lambda \) without changing its direction (or reverses it if \( \lambda < 0 \)). Eigenvectors are defined up to a scalar multiple, and each eigenvalue may have multiple eigenvectors forming a subspace.

### Finding Eigenvalues
To find eigenvalues, solve the **characteristic equation**:

\[
\det(A - \lambda I) = 0
\]

where \( I \) is the \( n \times n \) identity matrix. The resulting polynomial (degree \( n \)) has up to \( n \) roots (eigenvalues), which may be real or complex and include multiplicities.

For a 2x2 matrix \( A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \):

\[
A - \lambda I = \begin{bmatrix} a - \lambda & b \\ c & d - \lambda \end{bmatrix}
\]

\[
\det(A - \lambda I) = (a - \lambda)(d - \lambda) - bc = \lambda^2 - (a + d)\lambda + (ad - bc) = 0
\]

Solve this quadratic equation for \( \lambda \).

### Finding Eigenvectors
For each eigenvalue \( \lambda \), solve the system:

\[
(A - \lambda I) \mathbf{v} = \mathbf{0}
\]

The non-trivial solutions \( \mathbf{v} \) are the eigenvectors corresponding to \( \lambda \).

### Geometric Intuition
Eigenvectors represent directions that remain invariant (up to scaling) under the transformation \( A \). Eigenvalues indicate the scaling factor:
- \( \lambda > 1 \): Stretches the eigenvector.
- \( 0 < \lambda < 1 \): Shrinks the eigenvector.
- \( \lambda = 0 \): Collapses to the origin.
- \( \lambda < 0 \): Reverses direction.
- Complex \( \lambda \): Involves rotation (in 2D).

---

## ML Context: Why Eigenvalues and Eigenvectors Matter

Eigenvalues and eigenvectors are critical in ML:
- **Covariance and PCA**: The eigenvectors of a covariance matrix define principal components (directions of maximum variance), and eigenvalues indicate their importance.
- **Stability Analysis**: In dynamical systems or optimization, eigenvalues of system matrices determine stability (e.g., convergence in gradient descent).
- **Spectral Clustering**: Eigenvectors of graph Laplacians partition data into clusters.
- **Neural Networks**: Eigenvalues of weight matrices influence training dynamics and generalization.

Understanding these concepts enables you to analyze data structure, reduce dimensionality, and optimize algorithms.

---

## Python Code: Eigenvalues and Eigenvectors

Let’s compute eigenvalues and eigenvectors using **NumPy** and **PyTorch**, with visualizations to illustrate their geometric meaning.

### Setup
Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Computing Eigenvalues and Eigenvectors
Let’s compute for a 2x2 matrix:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix
A = np.array([
    [2, 1],
    [1, 2]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Print results
print("Matrix A:\n", A)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)
```

**Output:**
```
Matrix A:
 [[2 1]
  [1 2]]

Eigenvalues: [3. 1.]

Eigenvectors (columns):
 [[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]
```

The eigenvalues are \( \lambda_1 = 3 \), \( \lambda_2 = 1 \), with eigenvectors approximately \( [0.707, 0.707] \) and \( [-0.707, 0.707] \), corresponding to directions along \( y = x \) and \( y = -x \).

### Geometric Visualization
Let’s visualize the transformation:

```python
# Visualize eigenvectors and their transformations
def plot_eigenvectors(A, eigenvalues, eigenvectors):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    
    # Plot original and transformed eigenvectors
    colors = ['blue', 'red']
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        Av = A @ v
        plt.quiver(*origin, *v, color=colors[i], scale=1, scale_units='xy', angles='xy', alpha=0.5)
        plt.quiver(*origin, *Av, color=colors[i], scale=1, scale_units='xy', angles='xy')
        plt.text(v[0], v[1], f'v{i+1}', color=colors[i], fontsize=12)
        plt.text(Av[0], Av[1], f'Av{i+1}', color=colors[i], fontsize=12)
    
    plt.grid(True)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Eigenvectors and Their Transformations')
    plt.show()

plot_eigenvectors(A, eigenvalues, eigenvectors)
```

This plots the eigenvectors \( \mathbf{v}_1, \mathbf{v}_2 \) (blue, red) and their transformations \( A \mathbf{v}_1, A \mathbf{v}_2 \). Since \( A \mathbf{v}_1 = 3 \mathbf{v}_1 \) and \( A \mathbf{v}_2 = 1 \mathbf{v}_2 \), the transformed vectors are scaled versions along the same directions.

### Verifying Eigenproperties
Let’s verify \( A \mathbf{v} = \lambda \mathbf{v} \):

```python
# Verify eigenvalue equation
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lambda_v = eigenvalues[i] * v
    print(f"\nEigenvector {i+1}:", v)
    print(f"A @ v_{i+1}:", Av)
    print(f"λ_{i+1} * v_{i+1}:", lambda_v)
    print(f"Match?", np.allclose(Av, lambda_v))
```

**Output:**
```
Eigenvector 1: [0.70710678 0.70710678]
A @ v_1: [2.12132034 2.12132034]
λ_1 * v_1: [2.12132034 2.12132034]
Match? True

Eigenvector 2: [-0.70710678  0.70710678]
A @ v_2: [-0.70710678  0.70710678]
λ_2 * v_2: [-0.70710678  0.70710678]
Match? True
```

This confirms the eigenvalue equation holds.

### PyTorch: Eigenvalues
Let’s compute eigenvalues in PyTorch:

```python
import torch

# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute eigenvalues
eigenvalues_torch = torch.linalg.eigvals(A_torch)

# Print results
print("PyTorch eigenvalues:", eigenvalues_torch.numpy())
```

**Output:**
```
PyTorch eigenvalues: [3.+0.j 1.+0.j]
```

PyTorch returns complex eigenvalues, but here they are real and match NumPy’s.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be discussed in the next post!

1. **Eigenvalues**: Create a \( 3 \times 3 \) matrix with random integers between -5 and 5. Compute its eigenvalues using NumPy.
2. **Eigenvectors**: For the matrix in Exercise 1, compute its eigenvectors and verify \( A \mathbf{v} = \lambda \mathbf{v} \) for one eigenvalue-eigenvector pair.
3. **PyTorch Eigenvalues**: Convert the matrix from Exercise 1 to a PyTorch tensor, compute its eigenvalues, and verify they match NumPy’s.
4. **Geometric Transformation**: Define a \( 2 \times 2 \) matrix (e.g., \( \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix} \)). Plot its eigenvectors and their transformations under \( A \).
5. **Covariance Matrix**: Create a \( 5 \times 2 \) matrix of 5 2D points, compute the covariance matrix, and find its eigenvalues and eigenvectors to identify principal directions.
6. **Stability Check**: Create a \( 2 \times 2 \) matrix and compute its eigenvalues. Determine if the system is stable (all eigenvalues have absolute value < 1).

---

## What’s Next?

In the next post, we’ll explore **Singular Value Decomposition (SVD)**, a cornerstone for dimensionality reduction and noise filtering. We’ll provide more Python code and exercises to continue building your ML expertise.

Happy learning, and see you in Part 11!
