+++
title = "Linear Algebra for Machine Learning, Part 7: Orthogonality and Projections"
author = "Artintellica"
date = "2025-06-29"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0026-linalg-07-orthogonality"
+++

# Linear Algebra for Machine Learning, Part 7: Orthogonality and Projections

Welcome to the seventh post in our series on **Linear Algebra for Machine
Learning**, kicking off Part II: Core Theorems and Algorithms! After exploring
norms and distances, we now dive into **orthogonality** and **projections**,
fundamental concepts for error decomposition, principal component analysis
(PCA), and embeddings in machine learning (ML). In this post, we’ll cover the
mathematical foundations, their ML applications, and how to implement them in
Python using **NumPy** and **PyTorch**. We’ll include visualizations, the
Gram-Schmidt process, and Python exercises to solidify your understanding.

---

## The Math: Orthogonality and Projections

### Orthogonality

Two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ are **orthogonal** if
their dot product is zero:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = 0
$$

Geometrically, orthogonal vectors are perpendicular (form a 90° angle). A set of
vectors is **orthonormal** if they are pairwise orthogonal and each has a unit
length ($\|\mathbf{v}_i\|_2 = 1$). For an orthonormal set
$\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_k\}$:

$$
\mathbf{u}_i \cdot \mathbf{u}_j = \begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
$$

### Projections

The **projection** of a vector $\mathbf{u}$ onto a vector $\mathbf{v}$ is the
vector along $\mathbf{v}$’s direction that is closest to $\mathbf{u}$:

$$
\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}
$$

(**Note**: The subscript 2 in $\|\mathbf{v}\|_2$ indicates the L2 norm, which is
the Euclidean norm. [See more on norms](/blog/0025-linalg-06-norms.md).)

If $\mathbf{v}$ is a unit vector ($\|\mathbf{v}\|_2 = 1$), this simplifies to:

$$
\text{proj}_{\mathbf{v}} \mathbf{u} = (\mathbf{u} \cdot \mathbf{v}) \mathbf{v}
$$

The vector $\mathbf{u} - \text{proj}_{\mathbf{v}} \mathbf{u}$ is orthogonal to
$\mathbf{v}$, enabling error decomposition.

### Gram-Schmidt Process

The **Gram-Schmidt process** transforms a set of linearly independent vectors
$\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ into an orthonormal set
$\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_k\}$:

1. Start with $\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|_2}$.
2. For each $i = 2, \dots, k$:

   - Compute the orthogonal vector:

     $$
     \mathbf{w}_i = \mathbf{v}_i - \sum_{j=1}^{i-1} (\mathbf{v}_i \cdot \mathbf{u}_j) \mathbf{u}_j
     $$

   - Normalize: $\mathbf{u}_i = \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|_2}$.

This produces an orthonormal basis for the span of the original vectors.

---

## ML Context: Why Orthogonality and Projections Matter

Orthogonality and projections are central to ML:

- **Error Decomposition**: Projections split data into components along a
  direction (e.g., a model’s prediction) and orthogonal errors, used in least
  squares and regression.
- **PCA**: Principal components are orthogonal directions of maximum variance,
  found via projections.
- **Embeddings**: Orthogonal bases in embeddings (e.g., word vectors) ensure
  efficient, non-redundant representations.
- **Optimization**: Orthogonal gradients in training improve convergence in
  algorithms like gradient descent.

These concepts enable efficient and interpretable ML models.

---

## Python Code: Orthogonality and Projections

Let’s implement orthogonality checks, projections, and the Gram-Schmidt process
using **NumPy** and **PyTorch**, with visualizations to illustrate their
geometry.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Orthogonality Check

Let’s verify orthogonality for two vectors:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
u = np.array([1, 0])
v = np.array([0, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
print("Orthogonal?", np.isclose(dot_product, 0, atol=1e-10))

# Visualize vectors
def plot_2d_vectors(vectors, labels, colors, title):
    plt.figure(figsize=(6, 6))
    origin = np.zeros(2)
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(*origin, *vec, color=color, scale=1, scale_units='xy', angles='xy')
        plt.text(vec[0], vec[1], label, color=color, fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

plot_2d_vectors(
    [u, v],
    ['u', 'v'],
    ['blue', 'red'],
    "Orthogonal Vectors"
)
```

**Output:**

```
Vector u: [1 0]
Vector v: [0 1]
Dot product u · v: 0
Orthogonal? True
```

This confirms $\mathbf{u}$ and $\mathbf{v}$ are orthogonal (dot product = 0) and
visualizes their perpendicularity.

### Projection

Let’s project a vector onto another:

```python
# Define vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute projection of u onto v
dot_uv = np.dot(u, v)
norm_v_squared = np.sum(v**2)
projection = (dot_uv / norm_v_squared) * v

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Projection of u onto v:", projection)

# Visualize
plot_2d_vectors(
    [u, v, projection],
    ['u', 'v', 'proj_v(u)'],
    ['blue', 'red', 'green'],
    "Projection of u onto v"
)
```

**Output:**

```
Vector u: [1 2]
Vector v: [3 1]
Projection of u onto v: [1.5 0.5]
```

This computes
$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|_2^2} \mathbf{v}$,
projecting $\mathbf{u}$ onto $\mathbf{v}$, and plots the result along
$\mathbf{v}$’s direction.

### Gram-Schmidt Process

Let’s apply Gram-Schmidt to create an orthonormal basis:

```python
# Define two linearly independent vectors
v1 = np.array([1, 1])
v2 = np.array([1, 0])

# Gram-Schmidt process
u1 = v1 / np.linalg.norm(v1)  # Normalize v1
w2 = v2 - np.dot(v2, u1) * u1  # Orthogonalize v2
u2 = w2 / np.linalg.norm(w2)  # Normalize w2

# Verify orthonormality
print("Orthonormal basis:")
print("u1:", u1)
print("u2:", u2)
print("u1 · u2:", np.dot(u1, u2))
print("Norm u1:", np.linalg.norm(u1))
print("Norm u2:", np.linalg.norm(u2))

# Visualize
plot_2d_vectors(
    [v1, v2, u1, u2],
    ['v1', 'v2', 'u1', 'u2'],
    ['blue', 'red', 'green', 'purple'],
    "Gram-Schmidt Orthonormal Basis"
)
```

**Output:**

```
Orthonormal basis:
u1: [0.70710678 0.70710678]
u2: [ 0.70710678 -0.70710678]
u1 · u2: 0.0
Norm u1: 1.0
Norm u2: 1.0
```

This applies Gram-Schmidt to $\mathbf{v}_1, \mathbf{v}_2$, producing orthonormal
vectors $\mathbf{u}_1, \mathbf{u}_2$, and verifies orthogonality (dot product
≈ 0) and unit length.

### PyTorch: Projection

Let’s compute a projection in PyTorch:

```python
import torch

# Convert to PyTorch tensors
u_torch = torch.tensor([1.0, 2.0])
v_torch = torch.tensor([3.0, 1.0])

# Compute projection
dot_uv_torch = torch.dot(u_torch, v_torch)
norm_v_squared_torch = torch.sum(v_torch**2)
projection_torch = (dot_uv_torch / norm_v_squared_torch) * v_torch

# Print result
print("PyTorch projection of u onto v:", projection_torch.numpy())
```

**Output:**

```
PyTorch projection of u onto v: [1.5 0.5]
```

This confirms PyTorch’s projection matches NumPy’s.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Orthogonality Check**: Create three 2D vectors with random integers between
   -5 and 5 using NumPy. Check pairwise orthogonality by computing dot products.
2. **Projection Calculation**: Compute the projection of $\mathbf{u} = [2, 3]$
   onto $\mathbf{v} = [1, 1]$ using NumPy. Visualize $\mathbf{u}$, $\mathbf{v}$,
   and the projection in 2D.
3. **PyTorch Projection**: Convert the vectors from Exercise 2 to PyTorch
   tensors, compute the projection, and verify it matches NumPy’s.
4. **Gram-Schmidt Process**: Apply Gram-Schmidt to two 2D vectors of your choice
   (e.g., $[2, 1]$, $[1, 2]$). Verify the resulting vectors are orthonormal and
   plot them.
5. **Error Decomposition**: Compute the projection of $\mathbf{u} = [1, 2]$ onto
   $\mathbf{v} = [3, 1]$, then find the orthogonal error vector
   $\mathbf{u} - \text{proj}_{\mathbf{v}} \mathbf{u}$. Verify it’s orthogonal to
   $\mathbf{v}$.
6. **Orthonormal Basis**: Create a $3 \times 3$ matrix with random integers.
   Apply Gram-Schmidt to its columns to form an orthonormal basis, and verify
   orthonormality.

---

## What’s Next?

In the next post, we’ll explore **matrix inverses and systems of equations**,
crucial for solving linear models and understanding backpropagation. We’ll
provide more Python code and exercises to keep building your ML expertise.

Happy learning, and see you in Part 8!
