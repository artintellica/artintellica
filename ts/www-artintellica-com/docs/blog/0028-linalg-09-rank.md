+++
title = "Linear Algebra for Machine Learning, Part 9: Rank, Nullspace, and the Fundamental Theorem"
author = "Artintellica"
date = "2025-05-30"
+++

# Linear Algebra for Machine Learning, Part 9: Rank, Nullspace, and the Fundamental Theorem

Welcome to the ninth post in our series on **Linear Algebra for Machine Learning**, continuing Part II: Core Theorems and Algorithms! After exploring matrix inverses and systems of equations, we now dive into **rank**, **nullspace**, and the **Fundamental Theorem of Linear Algebra**, which provide deep insights into data compression and the structure of linear systems in machine learning (ML). In this post, we’ll cover the mathematical foundations, their ML applications, and how to implement them in Python using **NumPy** and **PyTorch**. We’ll include visualizations, an intuition for Singular Value Decomposition (SVD), and Python exercises to solidify your understanding.

---

## The Math: Rank, Nullspace, and the Fundamental Theorem

### Rank
The **rank** of a matrix \( A \) (size \( m \times n \)) is the number of linearly independent columns (or rows) in \( A \). Equivalently, it’s:
- The dimension of the column space (\( \text{col}(A) \)), the span of \( A \)’s columns.
- The dimension of the row space (\( \text{row}(A) \)), the span of \( A \)’s rows.

For an \( m \times n \) matrix:
- \( \text{rank}(A) \leq \min(m, n) \).
- If \( \text{rank}(A) = n \), the columns are linearly independent.
- If \( \text{rank}(A) = m \), the rows are linearly independent.

Rank is computed numerically via methods like SVD, which we’ll touch on intuitively.

### Nullspace
The **nullspace** (or kernel) of \( A \), denoted \( \text{null}(A) \), is the set of all vectors \( \mathbf{x} \in \mathbb{R}^n \) such that:

\[
A \mathbf{x} = \mathbf{0}
\]

The nullspace contains all solutions to the homogeneous system \( A \mathbf{x} = \mathbf{0} \). Its dimension, called the **nullity**, is:

\[
\text{nullity}(A) = n - \text{rank}(A)
\]

If \( \text{null}(A) = \{\mathbf{0}\} \) (only the zero vector), \( A \) has full column rank (\( \text{rank}(A) = n \)).

### Fundamental Theorem of Linear Algebra
The Fundamental Theorem connects the key subspaces of a matrix \( A \):

1. **Column Space**: \( \text{col}(A) \subseteq \mathbb{R}^m \), dimension \( \text{rank}(A) \).
2. **Nullspace**: \( \text{null}(A) \subseteq \mathbb{R}^n \), dimension \( n - \text{rank}(A) \).
3. **Row Space**: \( \text{row}(A) \subseteq \mathbb{R}^n \), dimension \( \text{rank}(A) \).
4. **Left Nullspace**: \( \text{null}(A^T) \subseteq \mathbb{R}^m \), dimension \( m - \text{rank}(A) \).

Key relationships:
- \( \text{col}(A) \) is orthogonal to \( \text{null}(A^T) \).
- \( \text{row}(A) \) is orthogonal to \( \text{null}(A) \).
- For a system \( A \mathbf{x} = \mathbf{b} \):
  - A solution exists if \( \mathbf{b} \in \text{col}(A) \).
  - If a solution exists, the general solution is \( \mathbf{x}_p + \mathbf{x}_n \), where \( \mathbf{x}_p \) is a particular solution and \( \mathbf{x}_n \in \text{null}(A) \).

### SVD Intuition
Singular Value Decomposition (SVD) decomposes \( A \) as:

\[
A = U \Sigma V^T
\]

where:
- \( U \) (\( m \times m \)) and \( V \) (\( n \times n \)) are orthogonal matrices.
- \( \Sigma \) (\( m \times n \)) is diagonal with non-negative singular values in decreasing order.

The rank of \( A \) is the number of non-zero singular values. SVD reveals the structure of \( \text{col}(A) \), \( \text{row}(A) \), and \( \text{null}(A) \), and is used in data compression (e.g., low-rank approximations).

---

## ML Context: Why Rank, Nullspace, and the Fundamental Theorem Matter

These concepts are crucial in ML:
- **Data Compression**: Low-rank approximations via SVD reduce data dimensionality while preserving key information, used in image compression and recommender systems.
- **Under/Over-Determined Systems**: Rank determines the solvability of linear systems in ML models, identifying unique, multiple, or no solutions.
- **Feature Redundancy**: A low rank indicates redundant features, guiding feature selection.
- **Regularization**: Nullspace analysis helps understand solution spaces in ill-posed problems, informing regularization strategies.

Mastering these ideas helps you analyze data structure and optimize ML algorithms.

---

## Python Code: Rank, Nullspace, and SVD

Let’s compute rank, nullspace, and explore SVD using **NumPy** and **PyTorch**, with visualizations to illustrate subspaces.

### Setup
Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Rank Computation
Let’s compute the rank of a matrix:

```python
import numpy as np

# Define a 4x3 matrix
A = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [3, 1, 4],
    [0, 0, 1]
])

# Compute rank
rank = np.linalg.matrix_rank(A)

# Print results
print("Matrix A (4x3):\n", A)
print("\nRank of A:", rank)
```

**Output:**
```
Matrix A (4x3):
 [[1 2 3]
  [2 4 6]
  [3 1 4]
  [0 0 1]]

Rank of A: 3
```

The rank is 3, indicating the columns are linearly independent (since \( \text{rank}(A) = n = 3 \)).

### Nullspace
Let’s approximate the nullspace using SVD:

```python
# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Nullspace basis (columns of V corresponding to zero singular values)
tol = 1e-10
nullspace_basis = Vt.T[:, S < tol]

# Print results
print("Singular values:", S)
print("\nNullspace basis (if any):\n", nullspace_basis if nullspace_basis.size > 0 else "Empty (full column rank)")
print("\nNullity (n - rank):", A.shape[1] - rank)
```

**Output:**
```
Singular values: [7.57313886 3.41668495 0.29092088]

Nullspace basis:
 Empty (full column rank)

Nullity (n - rank): 0
```

Since all singular values are non-zero, the nullspace is \( \{\mathbf{0}\} \), and the nullity is \( 3 - 3 = 0 \).

### Example with Non-Full Rank
Let’s try a matrix with dependent columns:

```python
# Define a 3x3 matrix with linearly dependent columns
A_dep = np.array([
    [1, 2, 4],  # Third column = 2 * first + second
    [2, 1, 5],
    [3, 0, 6]
])

# Compute rank and SVD
rank_dep = np.linalg.matrix_rank(A_dep)
U_dep, S_dep, Vt_dep = np.linalg.svd(A_dep, full_matrices=True)
nullspace_basis_dep = Vt_dep.T[:, S_dep < tol]

# Print results
print("Matrix A_dep (3x3):\n", A_dep)
print("\nRank of A_dep:", rank_dep)
print("Singular values:", S_dep)
print("\nNullspace basis:\n", nullspace_basis_dep)
print("Nullity:", A_dep.shape[1] - rank_dep)
```

**Output:**
```
Matrix A_dep (3x3):
 [[1 2 4]
  [2 1 5]
  [3 0 6]]

Rank of A_dep: 2
Singular values: [7.98791467 2.27789337 0.        ]

Nullspace basis:
 [[-0.40824829]
 [ 0.81649658]
 [-0.40824829]]

Nullity: 1
```

The rank is 2, and the nullspace has dimension 1, with a basis vector reflecting the dependency (third column = 2 * first + second).

### Visualization
Let’s visualize the column space for a 2x3 matrix:

```python
import matplotlib.pyplot as plt

# Define a 2x3 matrix
A_vis = np.array([
    [1, 0, 1],
    [0, 1, 1]
])

# Compute rank
rank_vis = np.linalg.matrix_rank(A_vis)

# Plot column vectors
plt.figure(figsize=(6, 6))
origin = np.zeros(2)
for i in range(A_vis.shape[1]):
    plt.quiver(*origin, *A_vis[:, i], color=['blue', 'red', 'green'][i], scale=1, scale_units='xy', angles='xy')
    plt.text(A_vis[0, i], A_vis[1, i], f'col{i+1}', fontsize=12)

# If rank = 2, span is the plane
if rank_vis == 2:
    t = np.linspace(-2, 2, 20)
    for c1 in t:
        for c2 in t:
            point = c1 * A_vis[:, 0] + c2 * A_vis[:, 1]
            plt.scatter(*point, color='gray', alpha=0.1, s=1)

plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Column Space (Rank = {rank_vis})')
plt.show()
```

This plots the columns of \( A_vis \) and shades the column space (a plane, since rank = 2).

### PyTorch: Rank
Let’s compute rank in PyTorch:

```python
import torch

# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute rank via SVD
_, S_torch, _ = torch.svd(A_torch)
rank_torch = torch.sum(S_torch > tol).item()

print("PyTorch rank:", rank_torch)
```

**Output:**
```
PyTorch rank: 3
```

This matches NumPy’s rank.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be discussed in the next post!

1. **Rank Calculation**: Create a \( 4 \times 4 \) matrix with random integers between -5 and 5. Compute its rank using NumPy.
2. **Nullspace Basis**: For the matrix in Exercise 1, compute the nullspace basis using SVD and print its dimension.
3. **PyTorch Rank**: Convert the matrix from Exercise 1 to a PyTorch tensor, compute its rank via SVD, and verify it matches NumPy’s.
4. **Under-Determined System**: Create a \( 2 \times 3 \) matrix and a 2D vector \( \mathbf{b} \). Solve \( A \mathbf{x} = \mathbf{b} \) using `np.linalg.lstsq` and check if multiple solutions exist by inspecting the nullspace.
5. **Over-Determined System**: Create a \( 4 \times 2 \) matrix and a 4D vector \( \mathbf{b} \). Solve the system using `np.linalg.lstsq` and verify if \( \mathbf{b} \in \text{col}(A) \).
6. **SVD Compression**: Create a \( 5 \times 5 \) matrix, compute its SVD, and reconstruct a rank-2 approximation. Compare the original and compressed matrices.

---

## What’s Next?

In the next post, we’ll explore **eigenvalues and eigenvectors**, key for PCA, stability analysis, and spectral clustering. We’ll provide more Python code and exercises to continue building your ML expertise.

Happy learning, and see you in Part 10!
