+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.9: Mini-Project—Visualizing and Transforming Data with Tensors"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Welcome to the hands-on mini-project of Module 1! In this project, you’ll bring
together all your new skills: tensor operations, linear transformations,
visualization, and practical data analysis—the real engine of RL and ML. Data
doesn’t come perfectly scaled or centered, and visualizing transformations
reveals insights critical for debugging and understanding both learning agents
and traditional models.

In this post you will:

- Load or generate a simple 2D dataset in PyTorch.
- Apply matrix transformations (rotation, scaling, etc.) and visualize their
  effects.
- Compute summary statistics (mean, covariance) using tensor math.
- Center (zero-mean) and normalize (unit variance) data—essential steps for many
  RL and ML pipelines.

Ready? Let’s turn numbers into pictures and see linear algebra in action!

---

## Mathematics: Data Transformation, Mean, and Covariance

Given a set of $N$ 2D data points $\{\mathbf{x}_i\}_{i=1}^N$, arranged as an
$N \times 2$ matrix $X$:

### Data Transformations

A linear transformation with matrix $A$ maps $X$ to $X_{\text{new}} = X A^T$ (or
$A X^T$ if you store points as columns). In ML, this rotates, stretches, or
shifts the data.

### Mean Vector

The **mean** of the dataset is:

$$
\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i
$$

### Covariance Matrix

The **covariance matrix** measures the spread and direction of the data:

$$
\text{Cov}(X) = \frac{1}{N} (X - \bar{\mathbf{x}})^T (X - \bar{\mathbf{x}})
$$

This $2 \times 2$ matrix tells you how the $x$ and $y$ dimensions vary together.

### Centering and Normalizing

- **Centering:** Subtract the mean from each data point.
- **Normalizing:** (Per-dimension) Divide by standard deviation to make variance
  $1$.

---

## Python Demonstrations

Let’s build a small end-to-end project!

### Demo 1: Load or Generate a Synthetic 2D Dataset

```python
import torch
import matplotlib.pyplot as plt

# Generate 2D Gaussian blobs
torch.manual_seed(42)
N = 200
mean = torch.tensor([2.0, -3.0])
cov = torch.tensor([[3.0, 1.2],
                    [1.2, 2.0]])
L = torch.linalg.cholesky(cov)
data = torch.randn(N, 2) @ L.T + mean

# Visualize original data
plt.scatter(data[:,0], data[:,1], alpha=0.6)
plt.title("Original 2D Data")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
```

---

### Demo 2: Apply Various Matrix Transformations and Visualize

```python
import math

def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ], dtype=torch.float32)

def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=torch.float32)

# Transform: rotate 45°, scale x=0.5, y=2
theta = math.radians(45)
R = rotation_matrix(theta)
S = scaling_matrix(0.5, 2.0)
transformed_data = (data @ S.T) @ R.T  # scale, then rotate

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(data[:,0], data[:,1], alpha=0.5, label='Original')
plt.legend(); plt.axis("equal"); plt.title("Original")
plt.subplot(1,2,2)
plt.scatter(transformed_data[:,0], transformed_data[:,1], alpha=0.5, label='Transformed', color='orange')
plt.legend(); plt.axis("equal"); plt.title("Transformed")
plt.tight_layout()
plt.show()
```

---

### Demo 3: Compute the Mean and Covariance of the Dataset

```python
# Mean vector
mean_vec: torch.Tensor = data.mean(dim=0)
print("Mean vector:\n", mean_vec)

# Centered data
data_centered: torch.Tensor = data - mean_vec

# Covariance matrix
cov_matrix: torch.Tensor = (data_centered.T @ data_centered) / data.shape[0]
print("Covariance matrix:\n", cov_matrix)
```

---

### Demo 4: Center and Normalize the Data with PyTorch

```python
# Center data
data_centered = data - mean_vec

# Normalize to unit std (per feature)
std_vec: torch.Tensor = data_centered.std(dim=0, unbiased=False)
data_normalized: torch.Tensor = data_centered / std_vec

# Visualize normalized data
plt.scatter(data_normalized[:,0], data_normalized[:,1], alpha=0.7)
plt.title("Centered & Normalized Data (Unit Std)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()

print("Standard deviations after normalization:", data_normalized.std(dim=0, unbiased=False))
```

---

## Exercises

Here’s your mini-project checklist—work through each for a practical deep-dive!

### **Exercise 1:** Load a Simple 2D Dataset (or Generate Synthetic Data)

- Generate 300 two-dimensional data points from a normal distribution (mean =
  [1, 4], covariance = [[2, 1], [1, 3]]).
- Plot the raw data.

### **Exercise 2:** Apply Various Matrix Transformations and Visualize Before/After

- Create a rotation matrix for 90°, and a scaling matrix (scale x by 2, y by
  0.5).
- Apply both (try scaling then rotating).
- Plot the original and transformed datasets side-by-side.

### **Exercise 3:** Compute the Mean and Covariance Matrix of the Dataset

- Calculate and print the mean and covariance of your dataset using torch
  operations.
- Interpret the covariance matrix (are features correlated?).

### **Exercise 4:** Center and Normalize the Data with PyTorch

- Subtract the mean from all points (centering).
- Divide by the standard deviation along each axis (normalizing).
- Show the scatterplot after normalization and check that variance is $1$ in
  each direction.

---

### **Sample Starter Code for Exercises**

```python
import torch
import math
import matplotlib.pyplot as plt

def rotation_matrix(theta: float) -> torch.Tensor:
    return torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ], dtype=torch.float32)

def scaling_matrix(sx: float, sy: float) -> torch.Tensor:
    return torch.tensor([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=torch.float32)

# EXERCISE 1
torch.manual_seed(0)
mean = torch.tensor([1.0, 4.0])
cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
L = torch.linalg.cholesky(cov)
data = torch.randn(300, 2) @ L.T + mean
plt.scatter(data[:,0], data[:,1], alpha=0.5)
plt.title("Raw Data")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.show()

# EXERCISE 2
theta = math.radians(90)
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
data_transf = (data @ S.T) @ R.T  # scale then rotate
plt.subplot(1,2,1)
plt.scatter(data[:,0], data[:,1], alpha=0.5, label='Original')
plt.axis("equal"); plt.title("Original")
plt.subplot(1,2,2)
plt.scatter(data_transf[:,0], data_transf[:,1], alpha=0.5, color='orange', label='Transformed')
plt.axis("equal"); plt.title("Transformed")
plt.tight_layout(); plt.show()

# EXERCISE 3
mean_vec = data.mean(dim=0)
print("Mean vector:", mean_vec)
centered = data - mean_vec
cov_mat = (centered.T @ centered) / data.shape[0]
print("Covariance matrix:\n", cov_mat)

# EXERCISE 4
std_vec = centered.std(dim=0, unbiased=False)
normalized = centered / std_vec
plt.scatter(normalized[:,0], normalized[:,1], alpha=0.7)
plt.title("Centered and Normalized Data")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.show()
print("Std after normalization:", normalized.std(dim=0, unbiased=False))
```

---

## Conclusion

Congratulations on completing this mini-project! You have:

- Created and visualized real 2D datasets in PyTorch.
- Transformed data using the core tools of linear algebra.
- Calculated and interpreted statistical properties like mean and covariance.
- Practiced centering and normalization—a must for robust RL and neural network
  performance.

**Next up:** You’ll transition from linear algebra to optimization—learning
about gradients, loss surfaces, and taking your first learning steps toward
training models!

_Great work—keep experimenting, and see you in Module 2!_
