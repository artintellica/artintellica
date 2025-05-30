+++
title = "Linear Algebra for Machine Learning, Part 8: Matrix Inverses and Systems of Equations"
author = "Artintellica"
date = "2025-05-30"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0027-linalg-08-matrix-inverses"
+++

Welcome to the eighth post in our series on **Linear Algebra for Machine
Learning**, continuing Part II: Core Theorems and Algorithms! After exploring
orthogonality and projections, we now tackle **matrix inverses** and **systems
of linear equations**, critical tools for solving for model parameters and
understanding backpropagation in machine learning (ML). In this post, we’ll
cover the mathematical foundations, their ML applications, and how to implement
them in Python using **NumPy** and **PyTorch**. We’ll include visualizations and
Python exercises to reinforce your understanding.

---

## The Math: Matrix Inverses and Systems of Equations

### Matrix Inverses

A square matrix $A$ (size $n \times n$) is **invertible** if there exists a
matrix $A^{-1}$ such that:

$$
A A^{-1} = A^{-1} A = I
$$

where $I$ is the $n \times n$ identity matrix (1s on the diagonal, 0s
elsewhere). The inverse $A^{-1}$ “undoes” the transformation represented by $A$.
A matrix is invertible if and only if:

- It has full rank ($\text{rank}(A) = n$).
- Its determinant is non-zero ($\det(A) \neq 0$).
- Its columns (or rows) are linearly independent.

For a 2x2 matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse
is:

$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}, \quad \text{where} \quad \det(A) = ad - bc
$$

For larger matrices, computing the inverse involves methods like Gaussian
elimination, but we’ll use NumPy for practical computation.

### Systems of Linear Equations

A system of linear equations can be written as:

$$
A \mathbf{x} = \mathbf{b}
$$

where $A$ is an $n \times n$ coefficient matrix, $\mathbf{x}$ is the vector of
unknowns, and $\mathbf{b}$ is the constant vector. If $A$ is invertible, the
solution is:

$$
\mathbf{x} = A^{-1} \mathbf{b}
$$

However, solving directly with $A^{-1}$ can be numerically unstable for large
matrices. Instead, methods like LU decomposition (used by `np.linalg.solve`) are
more efficient and stable.

For example, consider:

$$
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \mathbf{x} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}
$$

We solve for $\mathbf{x} = [x_1, x_2]$ using the inverse or a solver.

### Invertibility and ML

In ML, invertibility ensures unique solutions for parameters (e.g., in linear
regression). Non-invertible matrices indicate redundant features or insufficient
data, requiring regularization or alternative methods.

---

## ML Context: Why Matrix Inverses and Systems Matter

Matrix inverses and linear systems are vital in ML:

- **Linear Regression**: The normal equations
  $(X^T X) \mathbf{w} = X^T \mathbf{y}$ solve for weights $\mathbf{w}$, often
  requiring $X^T X$ to be invertible.
- **Backpropagation**: Gradients in neural networks involve solving linear
  systems to update weights, leveraging matrix operations.
- **Optimization**: Inverses appear in second-order methods like Newton’s
  method, approximating Hessians.
- **Data Analysis**: Solving systems helps fit models to data, ensuring unique
  parameter estimates.

Understanding these concepts enables you to solve for model parameters
efficiently and diagnose issues like multicollinearity.

---

## Python Code: Matrix Inverses and Systems

Let’s compute matrix inverses and solve linear systems using **NumPy** and
**PyTorch**, with visualizations to illustrate solutions.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Matrix Inverse

Let’s compute the inverse of a 2x2 matrix:

```python
import numpy as np

# Define a 2x2 matrix
A = np.array([[2, 1],
              [1, 3]])

# Compute inverse
A_inv = np.linalg.inv(A)

# Verify A * A_inv = I
identity = A @ A_inv

# Print results
print("Matrix A:\n", A)
print("\nInverse A^-1:\n", A_inv)
print("\nA @ A^-1 (should be identity):\n", identity)
print("Is identity?", np.allclose(identity, np.eye(2)))
```

**Output:**

```
Matrix A:
 [[2 1]
  [1 3]]

Inverse A^-1:
 [[ 0.6 -0.2]
 [-0.2  0.4]]

A @ A^-1 (should be identity):
 [[1. 0.]
 [0. 1.]]

Is identity? True
```

This computes $A^{-1}$ and verifies $A A^{-1} \approx I$, confirming
correctness.

### Solving a Linear System

Let’s solve $A \mathbf{x} = \mathbf{b}$:

```python
# Define b
b = np.array([5, 4])

# Solve using np.linalg.solve
x = np.linalg.solve(A, b)

# Solve using inverse (for comparison)
x_inv = A_inv @ b

# Print results
print("Matrix A:\n", A)
print("Vector b:", b)
print("Solution x (np.linalg.solve):", x)
print("Solution x (A^-1 @ b):", x_inv)
print("Solutions match?", np.allclose(x, x_inv))
```

**Output:**

```
Matrix A:
 [[2 1]
  [1 3]]
Vector b: [5 4]
Solution x (np.linalg.solve): [2.2 0.6]
Solution x (A^-1 @ b): [2.2 0.6]
Solutions match? True
```

This solves
$\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \mathbf{x} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}$,
yielding $\mathbf{x} = [2.2, 0.6]$, using both `np.linalg.solve` and the inverse
method.

### Visualization

Let’s visualize the system as intersecting lines:

```python
import matplotlib.pyplot as plt

# Define lines: 2x + y = 5, x + 3y = 4
x_vals = np.linspace(-1, 4, 100)
y1 = (5 - 2 * x_vals)  # From 2x + y = 5
y2 = (4 - x_vals) / 3  # From x + 3y = 4

# Plot
plt.figure(figsize=(6, 6))
plt.plot(x_vals, y1, label='2x + y = 5', color='blue')
plt.plot(x_vals, y2, label='x + 3y = 4', color='red')
plt.scatter(x[0], x[1], color='green', s=100, label='Solution')
plt.text(x[0], x[1], f'({x[0]:.1f}, {x[1]:.1f})', color='green', fontsize=12)
plt.grid(True)
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution to Linear System')
plt.legend()
plt.show()
```

This plots the two equations as lines, with their intersection at the solution
$(2.2, 0.6)$.

### PyTorch: Solving a System

Let’s solve the system in PyTorch:

```python
import torch

# Convert to PyTorch tensors
A_torch = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
b_torch = torch.tensor([5.0, 4.0])

# Solve using torch.linalg.solve
x_torch = torch.linalg.solve(A_torch, b_torch)

# Print result
print("PyTorch solution x:", x_torch.numpy())
```

**Output:**

```
PyTorch solution x: [2.2 0.6]
```

This confirms PyTorch’s solution matches NumPy’s.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Matrix Inverse**: Create a $2 \times 2$ matrix with random integers between
   1 and 5. Compute its inverse using NumPy and verify $A A^{-1} = I$.
2. **Linear System**: Define a $3 \times 3$ matrix and a 3D vector $\mathbf{b}$.
   Solve $A \mathbf{x} = \mathbf{b}$ using `np.linalg.solve` and the inverse
   method, and verify the solutions match.
3. **PyTorch System**: Convert the matrix and vector from Exercise 2 to PyTorch
   tensors, solve the system, and verify it matches NumPy’s.
4. **Non-Invertible Matrix**: Create a $3 \times 3$ matrix with linearly
   dependent columns (e.g., one column is a multiple of another). Attempt to
   compute its inverse and handle the resulting error.
5. **Visualization**: Solve a $2 \times 2$ system of your choice and plot the
   lines and solution point, similar to the example above.
6. **Determinant and Invertibility**: Create a $3 \times 3$ matrix with random
   integers. Compute its determinant and check if it’s invertible. If
   invertible, compute the inverse.

---

## What’s Next?

In the next post, we’ll explore **rank, nullspace, and the fundamental theorem
of linear algebra**, key for data compression and understanding system
solutions. We’ll provide more Python code and exercises to continue building
your ML expertise.

Happy learning, and see you in Part 9!
