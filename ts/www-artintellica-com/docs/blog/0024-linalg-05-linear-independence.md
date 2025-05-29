+++
title = "Linear Algebra for Machine Learning, Part 5: Linear Independence and Span"
author = "Artintellica"
date = "2025-05-29"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0024-linalg-05-linear-independence"
+++

Welcome to the fifth post in our series on **Linear Algebra for Machine
Learning**! After diving into dot products and cosine similarity, we now explore
**linear independence** and **span**, concepts that help us understand feature
redundancy and the expressive power of data representations in machine learning
(ML). In this post, we’ll cover the mathematical foundations, their relevance to
ML, and how to work with them in Python using **NumPy** and **PyTorch**. We’ll
include visualizations, a Gram matrix to assess independence, and Python
exercises to reinforce your understanding.

---

## The Math: Linear Independence and Span

### Linear Independence

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ in
$\mathbb{R}^n$ is **linearly independent** if no vector can be written as a
linear combination of the others. Mathematically, the only solution to:

$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}
$$

is $c_1 = c_2 = \dots = c_k = 0$, where $c_i$ are scalars. If any non-trivial
combination (i.e., some $c_i \neq 0$) yields the zero vector, the vectors are
**linearly dependent**, meaning at least one vector is redundant.

For example, in $\mathbb{R}^2$:

- $\mathbf{v}_1 = [1, 0]$, $\mathbf{v}_2 = [0, 1]$ are linearly independent (no
  scalar multiple of one equals the other).
- $\mathbf{v}_1 = [1, 2]$, $\mathbf{v}_2 = [2, 4]$ are linearly dependent
  ($\mathbf{v}_2 = 2 \mathbf{v}_1$).

### Span

The **span** of a set of vectors
$\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is the set of all possible
linear combinations:

$$
\text{span}(\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k) = \{ c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k \mid c_i \in \mathbb{R} \}
$$

Geometrically, the span is:

- A line if $k = 1$ (span of one vector).
- A plane if $k = 2$ and the vectors are linearly independent.
- The entire space $\mathbb{R}^n$ if $k \geq n$ and the vectors are linearly
  independent.

### Gram Matrix

The **Gram matrix** helps assess linear independence. For vectors
$\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k$, the Gram matrix $G$ is:

$$
G_{ij} = \mathbf{v}_i \cdot \mathbf{v}_j
$$

If $G$ is invertible (non-zero determinant), the vectors are linearly
independent. If not, they are linearly dependent.

---

## ML Context: Why Linear Independence and Span Matter

In machine learning:

- **Feature Redundancy**: Linearly dependent features (e.g., height in cm and
  meters) add no new information, wasting computational resources and risking
  overfitting. Linear independence ensures each feature contributes unique
  information.
- **Expressiveness**: The span of features determines the range of patterns a
  model can capture. A larger span (e.g., spanning $\mathbb{R}^n$) allows more
  expressive models, critical for tasks like regression or classification.
- **Dimensionality Reduction**: Techniques like PCA rely on identifying linearly
  independent directions to reduce redundant features.

Understanding these concepts helps you design efficient, expressive ML models.

---

## Python Code: Linear Independence and Span

Let’s explore linear independence and span using **NumPy** and **PyTorch**, with
visualizations and a Gram matrix to test independence.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Testing Linear Independence

Let’s create sets of vectors and check their independence using the Gram matrix:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two sets of 2D vectors
independent_vectors = np.array([[1, 0], [0, 1]])  # Linearly independent
dependent_vectors = np.array([[1, 2], [2, 4]])    # Linearly dependent

# Compute Gram matrices
gram_independent = independent_vectors.T @ independent_vectors
gram_dependent = dependent_vectors.T @ dependent_vectors

# Check determinants
det_independent = np.linalg.det(gram_independent)
det_dependent = np.linalg.det(gram_dependent)

# Print results
print("Independent vectors:\n", independent_vectors)
print("Gram matrix (independent):\n", gram_independent)
print("Determinant (independent):", det_independent)
print("\nDependent vectors:\n", dependent_vectors)
print("Gram matrix (dependent):\n", gram_dependent)
print("Determinant (dependent):", det_dependent)
```

**Output:**

```
Independent vectors:
 [[1 0]
  [0 1]]
Gram matrix (independent):
 [[1 0]
  [0 1]]
Determinant (independent): 1.0

Dependent vectors:
 [[1 2]
  [2 4]]
Gram matrix (dependent):
 [[ 5 10]
  [10 20]]
Determinant (dependent): 0.0
```

The non-zero determinant for the independent set confirms linear independence,
while the zero determinant for the dependent set indicates linear dependence.

### Visualizing Span

Let’s visualize the span of two vectors in 2D:

```python
# Visualize span of independent vectors
def plot_2d_vectors(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)

    # Plot span as a shaded region (for independent vectors)
    if len(vectors) == 2 and np.linalg.det(vectors) != 0:
        t = np.linspace(-10, 10, 100)
        for c1 in np.linspace(-2, 2, 20):
            for c2 in np.linspace(-2, 2, 20):
                point = c1 * vectors[0] + c2 * vectors[1]
                plt.scatter(point[0], point[1], color='gray', alpha=0.1, s=1)

    plt.grid(True)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

# Plot independent vectors
plot_2d_vectors(
    independent_vectors,
    ['v1', 'v2'],
    ['blue', 'red'],
    "Span of Linearly Independent Vectors"
)

# Plot dependent vectors (span is a line)
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, dependent_vectors[0, 0], dependent_vectors[0, 1], color='blue', scale=1, scale_units='xy', angles='xy')
plt.text(dependent_vectors[0, 0], dependent_vectors[0, 1], 'v1, v2', color='blue', fontsize=12)
t = np.linspace(-3, 3, 100)
line = t[:, np.newaxis] * dependent_vectors[0]
plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.5)
plt.grid(True)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Span of Linearly Dependent Vectors (Line)")
plt.show()
```

This visualizes:

- The span of $[1, 0]$ and $[0, 1]$ as the entire 2D plane (independent).
- The span of $[1, 2]$ and $[2, 4]$ as a line (dependent, since
  $[2, 4] = 2 \cdot [1, 2]$).

### PyTorch: Gram Matrix

Let’s compute a Gram matrix in PyTorch:

```python
import torch

# Convert to PyTorch tensors
ind_vectors_torch = torch.tensor(independent_vectors, dtype=torch.float32)

# Compute Gram matrix
gram_torch = ind_vectors_torch.T @ ind_vectors_torch

# Print result
print("PyTorch Gram matrix (independent):\n", gram_torch.numpy())
```

**Output:**

```
PyTorch Gram matrix (independent):
 [[1. 0.]
  [0. 1.]]
```

This confirms PyTorch’s Gram matrix matches NumPy’s.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Linear Independence Test**: Create three 2D vectors with random integers
   between -5 and 5 using NumPy. Compute the Gram matrix and its determinant to
   check if they are linearly independent (note: three 2D vectors are always
   dependent).
2. **Span Visualization**: Create two 2D vectors and visualize their span. If
   they are linearly independent, show the plane; if dependent, show the line.
3. **PyTorch Gram Matrix**: Convert the vectors from Exercise 1 to PyTorch
   tensors, compute the Gram matrix, and verify it matches NumPy’s.
4. **Feature Redundancy**: Create a $3 \times 3$ matrix representing 3 samples
   with 3 features. Check if the features (columns) are linearly independent
   using the Gram matrix.
5. **Linear Combination**: Create a set of two linearly independent 3D vectors.
   Compute a linear combination with random coefficients between -1 and 1, and
   plot the result in 2D (first two components).
6. **Rank and Independence**: Create a $4 \times 3$ matrix with random integers.
   Compute its rank using `np.linalg.matrix_rank` and determine if the columns
   are linearly independent.

---

## What’s Next?

In the next post, we’ll explore **norms and distances**, critical for loss
functions, regularization, and gradient scaling in ML. We’ll provide more Python
code and exercises to keep building your skills.

Happy learning, and see you in Part 6!
