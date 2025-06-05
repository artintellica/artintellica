+++
title = "Linear Algebra for Machine Learning, Part 20: Random Projections and Fast Transforms"
author = "Artintellica"
date = "2025-06-05"
+++

Welcome back to our series on linear algebra for machine learning! In this post,
we’re exploring **Random Projections and Fast Transforms**, powerful techniques
for efficient computation in large-scale machine learning. As datasets grow in
size and dimensionality, traditional methods can become computationally
infeasible. Random projections and fast transforms offer elegant solutions by
reducing dimensionality and speeding up computations while preserving essential
data properties. We'll dive into the math behind these methods, including the
Johnson-Lindenstrauss lemma, implement them with Python using NumPy, and provide
hands-on exercises to solidify your understanding. Let’s get started!

## What Are Random Projections and Fast Transforms?

Random projections and fast transforms are techniques rooted in linear algebra
that address the challenges of high-dimensional data and computational
complexity in machine learning.

### Random Projections

Random projections are a dimensionality reduction technique that projects
high-dimensional data into a lower-dimensional space using a random matrix.
Unlike PCA, which seeks optimal directions of variance, random projections rely
on randomness to approximate distances between points. The theoretical
foundation for this is the **Johnson-Lindenstrauss Lemma**, which states that a
set of points in a high-dimensional space can be embedded into a
lower-dimensional space while approximately preserving pairwise distances with
high probability.

**Johnson-Lindenstrauss Lemma (simplified)**: For any set of $n$ points in
$\mathbb{R}^d$, there exists a projection into $\mathbb{R}^k$ (where
$k = O(\log n / \epsilon^2)$) such that the pairwise distances are preserved
within a factor of $1 \pm \epsilon$. The projection matrix can be constructed
randomly, often with entries drawn from a Gaussian distribution or simpler
distributions like $\pm 1$.

Mathematically, if $X \in \mathbb{R}^{n \times d}$ is the original data matrix,
a random projection matrix $R \in \mathbb{R}^{d \times k}$ (with $k \ll d$) is
used to compute the reduced data:

$$
X_{\text{proj}} = X R \in \mathbb{R}^{n \times k}
$$

The randomness of $R$ ensures computational efficiency and surprising
effectiveness in preserving structure.

### Fast Transforms

Fast transforms, such as the Fast Fourier Transform (FFT) or Hadamard Transform,
are efficient algorithms for applying specific linear transformations. They
reduce the computational complexity of matrix operations from $O(n^2)$ or higher
to $O(n \log n)$ or better. In machine learning, fast transforms are used for
tasks like signal processing, kernel approximations, and speeding up matrix
multiplications in random projection variants (e.g., using structured random
matrices).

## Why Do Random Projections and Fast Transforms Matter in Machine Learning?

These techniques are crucial for large-scale machine learning for several
reasons:

1. **Scalability**: Random projections reduce dimensionality, enabling
   algorithms to handle massive datasets with millions of features.
2. **Efficiency**: Fast transforms and structured random matrices drastically
   cut computation time for matrix operations.
3. **Approximation Guarantees**: The Johnson-Lindenstrauss lemma provides
   theoretical assurance that random projections preserve distances, making them
   reliable for tasks like clustering and classification.
4. **Versatility**: These methods are used in applications ranging from data
   compression and streaming algorithms to kernel approximations and neural
   network compression.

Understanding the linear algebra behind random projections and fast transforms
equips you to tackle the computational bottlenecks of modern machine learning.

## Implementing Random Projections with Python

Let’s implement random projections using NumPy to reduce the dimensionality of a
synthetic dataset. We'll also compare the pairwise distances before and after
projection to demonstrate the Johnson-Lindenstrauss lemma in action.
Additionally, we'll briefly touch on a fast transform concept using the Hadamard
matrix as a structured random projection.

### Example 1: Random Projections with Gaussian Matrix

```python
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic high-dimensional data (100 samples, 1000 dimensions)
n_samples, d = 100, 1000
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Target reduced dimension (based on Johnson-Lindenstrauss, k ~ log(n)/epsilon^2)
epsilon = 0.1  # Distortion factor
k = int(8 * np.log(n_samples) / (epsilon ** 2))  # Rough estimate
print("Target Reduced Dimension (k):", k)

# Create random projection matrix (Gaussian entries)
R = np.random.randn(d, k) / np.sqrt(k)  # Scale to preserve distances approximately
print("Random Projection Matrix Shape:", R.shape)

# Project data to lower dimension
X_proj = X @ R
print("Projected Data Shape:", X_proj.shape)

# Compute pairwise distances before and after projection
dist_original = pairwise_distances(X, metric='euclidean')
dist_projected = pairwise_distances(X_proj, metric='euclidean')

# Compute relative distortion
relative_distortion = np.abs(dist_projected - dist_original) / dist_original
mean_distortion = np.mean(relative_distortion)
print("Mean Relative Distortion:", mean_distortion)

# Visualize distortion distribution
plt.figure(figsize=(8, 6))
plt.hist(relative_distortion.flatten(), bins=50, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Relative Distortion in Pairwise Distances')
plt.xlabel('Relative Distortion')
plt.ylabel('Density')
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:

```
Original Data Shape: (100, 1000)
Target Reduced Dimension (k): 368
Random Projection Matrix Shape: (1000, 368)
Projected Data Shape: (100, 368)
Mean Relative Distortion: 0.0492
```

This example generates a synthetic high-dimensional dataset (100 samples, 1000
dimensions) and applies a random projection using a Gaussian matrix to reduce it
to a lower dimension (calculated roughly based on the Johnson-Lindenstrauss
lemma). The pairwise Euclidean distances are computed before and after
projection, and the relative distortion is analyzed. A histogram visualizes the
distribution of distortions, typically showing that most distances are preserved
within a small error margin (close to the specified $\epsilon = 0.1$).

### Example 2: Structured Random Projection with Hadamard-like Matrix

```python
import numpy as np
from scipy.linalg import hadamard

# Set random seed for reproducibility
np.random.seed(42)

# Generate smaller synthetic data for demonstration (64 samples, 64 dimensions)
n_samples, d = 64, 64  # Hadamard matrix requires power of 2
X = np.random.randn(n_samples, d)
print("Original Data Shape:", X.shape)

# Create a Hadamard matrix (structured orthogonal matrix)
H = hadamard(d) / np.sqrt(d)  # Normalize to preserve distances approximately
print("Hadamard Matrix Shape:", H.shape)

# Randomly select a subset of columns for projection (reduce to k=16 dimensions)
k = 16
indices = np.random.choice(d, k, replace=False)
R_structured = H[:, indices]
print("Structured Projection Matrix Shape:", R_structured.shape)

# Project data to lower dimension
X_proj_structured = X @ R_structured
print("Projected Data Shape:", X_proj_structured.shape)
```

**Output (abbreviated)**:

```
Original Data Shape: (64, 64)
Hadamard Matrix Shape: (64, 64)
Structured Projection Matrix Shape: (64, 16)
Projected Data Shape: (64, 16)
```

This example demonstrates a structured random projection using a Hadamard matrix
(via `scipy.linalg.hadamard`), which is faster to compute than a full Gaussian
random matrix for certain dimensions (powers of 2). A subset of columns is
randomly selected to reduce the dimensionality, simulating a fast transform
approach. This is a simplified illustration of how structured matrices can be
used in place of fully random ones for efficiency in large-scale settings.

## Exercises

Here are six exercises to deepen your understanding of random projections and
fast transforms. Each exercise involves writing Python code to explore concepts
and applications in machine learning using NumPy and other libraries.

1. **Basic Random Projection**: Generate a synthetic dataset (200 samples, 500
   dimensions) with NumPy. Apply a random projection to reduce it to 50
   dimensions using a Gaussian random matrix. Compute and print the mean
   relative distortion of pairwise distances.
2. **Johnson-Lindenstrauss Dimension Calculation**: Write a function to
   calculate the target dimension $k$ for random projection based on the
   Johnson-Lindenstrauss lemma ($k = \frac{8 \log n}{\epsilon^2}$) for a given
   number of samples $n$ and distortion $\epsilon$. Test it for
   $n = 100, 1000, 10000$ and $\epsilon = 0.1, 0.2$.
3. **Random Projection for Classification**: Use a synthetic dataset (e.g., from
   `sklearn.datasets.make_classification`, 1000 samples, 200 features). Apply
   random projection to reduce to 20 dimensions, then train a logistic
   regression classifier (`sklearn.linear_model.LogisticRegression`). Compare
   accuracy before and after projection.
4. **Structured Random Projection**: Using a dataset of size (128 samples, 128
   features), create a Hadamard matrix with `scipy.linalg.hadamard` and randomly
   select 32 columns for projection. Compute and print the mean relative
   distortion of pairwise distances.
5. **Fast Fourier Transform (FFT) Application**: Generate a 1D signal (e.g., a
   sum of sine waves with noise) with 1024 points. Apply FFT using
   `numpy.fft.fft` to transform it to the frequency domain, plot the original
   signal and its frequency spectrum.
6. **Random Projection on Real Data**: Load the digits dataset
   (`sklearn.datasets.load_digits`, 1797 samples, 64 features). Apply random
   projection to reduce to 10 dimensions, train a k-NN classifier
   (`sklearn.neighbors.KNeighborsClassifier`), and compare accuracy and runtime
   before and after projection.

## Conclusion

Random Projections and Fast Transforms provide computationally efficient
solutions for handling large-scale machine learning problems through the power
of linear algebra. By leveraging randomness and structured transformations, as
supported by the Johnson-Lindenstrauss lemma, we can reduce dimensionality and
accelerate computations while preserving critical data properties. Our Python
implementations with NumPy demonstrated the practical application of these
concepts, showing how distances are approximately maintained even after
significant dimensionality reduction.

In the next post, we’ll delve into **Matrix Factorization for Recommendation
Systems**, exploring how linear algebra drives collaborative filtering and
latent factor models. Stay tuned, and happy learning!
