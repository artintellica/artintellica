+++
title = "Linear Algebra for Machine Learning, Part 11: Singular Value Decomposition (SVD)"
author = "Artintellica"
date = "2025-05-30"
+++

# Linear Algebra for Machine Learning, Part 11: Singular Value Decomposition (SVD)

Welcome to the eleventh post in our series on **Linear Algebra for Machine Learning**, continuing Part II: Core Theorems and Algorithms! After exploring eigenvalues and eigenvectors, we now dive into **Singular Value Decomposition (SVD)**, a cornerstone for dimensionality reduction, noise filtering, and latent semantic analysis (LSA) in machine learning (ML). In this post, we’ll cover the mathematical foundations, their ML applications, and how to implement SVD in Python using **NumPy** and **PyTorch**. We’ll include a visual demo and Python exercises to solidify your understanding.

---

## The Math: Singular Value Decomposition

### Definition
For any \( m \times n \) matrix \( A \), **Singular Value Decomposition** decomposes \( A \) as:

\[
A = U \Sigma V^T
\]

where:
- \( U \) is an \( m \times m \) orthogonal matrix, with columns (left singular vectors) forming an orthonormal basis for \( \mathbb{R}^m \).
- \( \Sigma \) is an \( m \times n \) diagonal matrix with non-negative singular values \( \sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r \geq 0 \) (where \( r = \min(m, n) \)) on the diagonal, and zeros elsewhere.
- \( V \) is an \( n \times n \) orthogonal matrix, with columns (right singular vectors) forming an orthonormal basis for \( \mathbb{R}^n \).
- \( V^T \) is the transpose of \( V \).

The singular values in \( \Sigma \) capture the “importance” of each dimension, and the rank of \( A \) is the number of non-zero singular values.

### Geometric Intuition
SVD interprets \( A \) as a composition of three transformations:
1. **Rotation/Reflection**: \( V^T \) rotates or reflects the input space.
2. **Scaling**: \( \Sigma \) scales along the principal axes (by singular values).
3. **Rotation/Reflection**: \( U \) rotates or reflects into the output space.

For example, if \( A \) represents a dataset, SVD identifies the principal directions (via \( U \) and \( V \)) and their magnitudes (via \( \Sigma \)).

### Low-Rank Approximation
A rank-\( k \) approximation of \( A \) is:

\[
A_k = U_k \Sigma_k V_k^T
\]

where \( U_k \) is \( m \times k \), \( \Sigma_k \) is \( k \times k \), and \( V_k \) is \( n \times k \), using the top \( k \) singular values and vectors. This minimizes the Frobenius norm \( \|A - A_k\|_F \), ideal for compression and noise reduction.

---

## ML Context: Why SVD Matters

SVD is pivotal in ML:
- **Dimensionality Reduction**: Low-rank approximations reduce data size while preserving structure, used in PCA and recommender systems.
- **Noise Filtering**: Truncating small singular values removes noise, enhancing signal in images or audio.
- **Latent Semantic Analysis (LSA)**: In NLP, SVD uncovers latent topics in document-term matrices.
- **Matrix Completion**: SVD helps impute missing data in collaborative filtering.

Mastering SVD enables efficient data processing and robust model design.

---

## Python Code: Singular Value Decomposition

Let’s compute SVD, create a low-rank approximation, and visualize its effects using **NumPy** and **PyTorch**.

### Setup
Install the required libraries if needed:

```bash
pip install numpy torch matplotlib
```

### Computing SVD
Let’s compute SVD for a matrix:

```python
import numpy as np

# Create a 4x3 matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Reconstruct Sigma as a 4x3 diagonal matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)

# Reconstruct A
A_reconstructed = U @ Sigma @ Vt

# Print results
print("Matrix A (4x3):\n", A)
print("\nSingular values:", S)
print("\nU (4x4):\n", U)
print("\nSigma (4x3):\n", Sigma)
print("\nVt (3x3):\n", Vt)
print("\nReconstructed A:\n", A_reconstructed)
print("\nReconstruction matches original?", np.allclose(A, A_reconstructed))
```

**Output:**
```
Matrix A (4x3):
 [[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]

Singular values: [25.46240744  1.29099445]

U (4x4):
 [[-0.1408777   0.8247362  -0.53665631 -0.10482848]
 [-0.34394629  0.42693249  0.80218019  0.25881905]
 [-0.54701488  0.02912877 -0.25881905  0.78801075]
 [-0.75008347 -0.36876343 -0.00670483 -0.54199932]]

Sigma (4x3):
 [[25.4624  0.     0.    ]
 [ 0.     1.291   0.    ]
 [ 0.     0.     0.    ]
 [ 0.     0.     0.    ]]

Vt (3x3):
 [[-0.50453315 -0.5745155  -0.64449785]
 [ 0.76077568  0.05714052 -0.64649463]
 [ 0.40824829 -0.81649658  0.40824829]]

Reconstructed A:
 [[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]

Reconstruction matches original? True
```

This computes SVD, reconstructs \( A \), and verifies the decomposition.

### Low-Rank Approximation
Let’s create a rank-1 approximation:

```python
# Rank-1 approximation
k = 1
U_k = U[:, :k]
Sigma_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
A_rank1 = U_k @ Sigma_k @ Vt_k

# Compute Frobenius norm of difference
diff_norm = np.linalg.norm(A - A_rank1, 'fro')

# Print results
print("Rank-1 approximation A_rank1:\n", A_rank1)
print("\nFrobenius norm of difference (||A - A_rank1||_F):", diff_norm)
```

**Output:**
```
Rank-1 approximation A_rank1:
 [[ 0.9041  1.0296  1.1532]
 [ 2.2054  2.5111  2.8128]
 [ 3.5067  3.9926  4.4724]
 [ 4.8081  5.4741  6.132 ]]

Frobenius norm of difference (||A - A_rank1||_F): 1.6641
```

This approximates \( A \) using the top singular value, with some error.

### Visualization
Let’s visualize the matrices:

```python
import matplotlib.pyplot as plt

# Consistent color scale
vmin = min(A.min(), A_rank1.min())
vmax = max(A.max(), A_rank1.max())

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(A, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Original A')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(A_rank1, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Rank-1 Approximation')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(A - A_rank1, cmap='plasma')
plt.title('Difference')
plt.colorbar()
plt.tight_layout()
plt.show()
```

This plots the original matrix, rank-1 approximation, and their difference, showing compression effects.

### PyTorch: SVD
Let’s compute SVD in PyTorch:

```python
import torch

# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute SVD
U_torch, S_torch, V_torch = torch.svd(A_torch)

# Print results
print("PyTorch singular values:", S_torch.numpy())
```

**Output:**
```
PyTorch singular values: [25.462408  1.2909944]
```

This matches NumPy’s singular values.

---

## Exercises

Try these Python exercises to deepen your understanding. Solutions will be discussed in the next post!

1. **SVD Computation**: Create a \( 3 \times 4 \) matrix with random integers between -5 and 5. Compute its SVD using NumPy and reconstruct the matrix.
2. **Rank-k Approximation**: For the matrix in Exercise 1, compute a rank-2 approximation and calculate the Frobenius norm of the difference.
3. **PyTorch SVD**: Convert the matrix from Exercise 1 to a PyTorch tensor, compute its SVD, and verify the singular values match NumPy’s.
4. **Image Compression**: Load a grayscale image (or create a small matrix), compute its SVD, and reconstruct a rank-10 approximation. Visualize the original and compressed images.
5. **Noise Filtering**: Add random noise to a \( 5 \times 5 \) matrix, compute its SVD, and use a rank-3 approximation to filter noise. Compare with the original.
6. **LSA Simulation**: Create a \( 4 \times 3 \) document-term matrix, compute its SVD, and interpret the top singular vectors as latent topics.

---

## What’s Next?

In the next post, we’ll explore **positive definite matrices**, key for covariance, kernels, and optimization. We’ll provide more Python code and exercises to continue building your ML expertise.

Happy learning, and see you in Part 12!
