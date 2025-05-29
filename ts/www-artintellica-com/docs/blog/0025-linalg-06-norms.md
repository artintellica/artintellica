+++
title = "Linear Algebra for Machine Learning, Part 6: Norms and Distances"
author = "Artintellica"
date = "2025-05-29"
+++

Welcome to the sixth post in our series on **Linear Algebra for Machine
Learning**! After exploring linear independence and span, we now turn to
**norms** and **distances**, essential tools for quantifying vector magnitudes
and differences between data points. These concepts are critical in machine
learning (ML) for loss functions, regularization, and gradient scaling. In this
post, we’ll cover their mathematical foundations, their applications in ML, and
how to implement them in Python using **NumPy** and **PyTorch**. We’ll include
visualizations and Python exercises to deepen your understanding.

---

## The Math: Norms and Distances

### Vector Norms

A **norm** measures the "size" or "length" of a vector. For a vector
$\mathbf{v} = [v_1, v_2, \dots, v_n] \in \mathbb{R}^n$, common norms include:

- **L1 Norm** (Manhattan norm):

  $$
  \|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|
  $$

  This sums the absolute values of the components, emphasizing sparsity in ML.

- **L2 Norm** (Euclidean norm):

  $$
  \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}
  $$

  This measures the straight-line distance from the origin, widely used in loss
  functions.

- **L∞ Norm** (Maximum norm):
  $$
  \|\mathbf{v}\|_\infty = \max_i |v_i|
  $$
  This takes the largest absolute component, useful in robustness analysis.

Norms satisfy properties like non-negativity ($\|\mathbf{v}\| \geq 0$),
scalability ($\|a\mathbf{v}\| = |a| \|\mathbf{v}\|$), and the triangle
inequality ($\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$).

### Distances

A **distance** measures how far apart two vectors are. For vectors
$\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, the distance is typically the norm of
their difference:

- **L1 Distance**:

  $$
  \|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^n |u_i - v_i|
  $$

- **L2 Distance** (Euclidean distance):

  $$
  \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}
  $$

- **L∞ Distance**:
  $$
  \|\mathbf{u} - \mathbf{v}\|_\infty = \max_i |u_i - v_i|
  $$

Distances are non-negative, symmetric
($d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$), and satisfy the
triangle inequality.

---

## ML Context: Why Norms and Distances Matter

Norms and distances are foundational in ML:

- **Loss Functions**: The L2 norm is used in mean squared error (MSE) to measure
  prediction errors, while the L1 norm appears in mean absolute error (MAE).
- **Regularization**: L1 regularization (Lasso) promotes sparsity in model
  weights, and L2 regularization (Ridge) prevents large weights to reduce
  overfitting.
- **Gradient Scaling**: Norms normalize gradients in optimization to ensure
  stable training.
- **Clustering and Similarity**: Distances like Euclidean or Manhattan are used
  in algorithms like k-means or nearest neighbors to group or compare data
  points.

Mastering these concepts helps you design robust ML models and evaluate their
performance.

---

## Python Code: Norms and Distances

Let’s compute L1, L2, and L∞ norms and distances using **NumPy** and
**PyTorch**, with visualizations to illustrate their geometric interpretations.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Vector Norms

Let’s compute norms for a vector:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a vector
v = np.array([3, 4])

# Compute norms
l1_norm = np.sum(np.abs(v))  # L1 norm
l2_norm = np.linalg.norm(v)  # L2 norm
linf_norm = np.max(np.abs(v))  # L∞ norm

# Print results
print("Vector v:", v)
print("L1 norm:", l1_norm)
print("L2 norm:", l2_norm)
print("L∞ norm:", linf_norm)

# Visualize vector
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, v[0], v[1], color='blue', scale=1, scale_units='xy', angles='xy')
plt.text(v[0], v[1], 'v', color='blue', fontsize=12)
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Vector for Norms")
plt.show()
```

**Output:**

```
Vector v: [3 4]
L1 norm: 7
L2 norm: 5.0
L∞ norm: 4
```

This computes the L1 ($|3| + |4| = 7$), L2 ($\sqrt{3^2 + 4^2} = 5$), and L∞
($\max(|3|, |4|) = 4$) norms for $\mathbf{v} = [3, 4]$, and plots the vector.

### Distances

Let’s compute distances between two vectors:

```python
# Define another vector
u = np.array([1, 1])

# Compute distances
l1_dist = np.sum(np.abs(u - v))
l2_dist = np.linalg.norm(u - v)
linf_dist = np.max(np.abs(u - v))

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("L1 distance:", l1_dist)
print("L2 distance:", l2_dist)
print("L∞ distance:", linf_dist)

# Visualize vectors and distance
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, u[0], u[1], color='red', scale=1, scale_units='xy', angles='xy')
plt.quiver(0, 0, v[0], v[1], color='blue', scale=1, scale_units='xy', angles='xy')
plt.plot([u[0], v[0]], [u[1], v[1]], 'g--', label='L2 distance')
plt.text(u[0], u[1], 'u', color='red', fontsize=12)
plt.text(v[0], v[1], 'v', color='blue', fontsize=12)
plt.grid(True)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Vectors and L2 Distance")
plt.legend()
plt.show()
```

**Output:**

```
Vector u: [1 1]
Vector v: [3 4]
L1 distance: 5
L2 distance: 3.605551275463989
L∞ distance: 3
```

This computes the L1 ($|1-3| + |1-4| = 5$), L2
($\sqrt{(1-3)^2 + (1-4)^2} \approx 3.606$), and L∞ ($\max(|1-3|, |1-4|) = 3$)
distances between $\mathbf{u} = [1, 1]$ and $\mathbf{v} = [3, 4]$. The plot
shows both vectors and a dashed line representing the L2 distance.

### PyTorch: Norms and Distances

Let’s compute norms and distances in PyTorch:

```python
import torch

# Convert to PyTorch tensors
u_torch = torch.tensor(u, dtype=torch.float32)
v_torch = torch.tensor(v, dtype=torch.float32)

# Compute norms for v
l1_norm_torch = torch.sum(torch.abs(v_torch))
l2_norm_torch = torch.norm(v_torch)
linf_norm_torch = torch.max(torch.abs(v_torch))

# Compute distances
l1_dist_torch = torch.sum(torch.abs(u_torch - v_torch))
l2_dist_torch = torch.norm(u_torch - v_torch)
linf_dist_torch = torch.max(torch.abs(u_torch - v_torch))

# Print results
print("PyTorch L1 norm (v):", l1_norm_torch.item())
print("PyTorch L2 norm (v):", l2_norm_torch.item())
print("PyTorch L∞ norm (v):", linf_norm_torch.item())
print("PyTorch L1 distance:", l1_dist_torch.item())
print("PyTorch L2 distance:", l2_dist_torch.item())
print("PyTorch L∞ distance:", linf_dist_torch.item())
```

**Output:**

```
PyTorch L1 norm (v): 7.0
PyTorch L2 norm (v): 5.0
PyTorch L∞ norm (v): 4.0
PyTorch L1 distance: 5.0
PyTorch L2 distance: 3.605551242828369
PyTorch L∞ distance: 3.0
```

This confirms PyTorch’s results match NumPy’s, with minor floating-point
differences.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be
discussed in the next post!

1. **Vector Norms**: Create a 3D vector with random integers between -5 and 5
   using NumPy. Compute its L1, L2, and L∞ norms and print the results.
2. **Distance Calculation**: Create two 2D vectors with random integers between
   -5 and 5. Compute their L1, L2, and L∞ distances and visualize the vectors
   with the L2 distance as a dashed line.
3. **PyTorch Norms**: Convert the vector from Exercise 1 to a PyTorch tensor and
   compute its L1, L2, and L∞ norms. Verify the results match NumPy’s.
4. **Regularization Effect**: Create a 2D vector and compute its L1 and L2
   norms. Scale the vector by 0.5 and recompute the norms to see the effect of
   regularization-like scaling.
5. **Nearest Neighbor**: Create a $5 \times 2$ matrix of 5 2D points and a query
   point. Compute the L2 distances to each point and find the closest one.
6. **Norm Comparison**: Generate 10 random 2D vectors and compute their L1 and
   L2 norms. Plot L1 vs. L2 norms in a scatter plot to compare their behavior.

---

## What’s Next?

In the next post, we’ll explore **orthogonality and projections**, key for error
decomposition, PCA, and embeddings. We’ll provide more Python code and exercises
to continue building your ML expertise.

Happy learning, and see you in Part 7!
