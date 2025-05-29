+++
title = "Linear Algebra for Machine Learning, Part 2: Matrices as Data & Transformations"
author = "Artintellica"
date = "2025-05-29"
+++

Welcome to the second post in our series on **Linear Algebra for Machine
Learning**! After exploring vectors, scalars, and vector spaces, we’re now
diving into **matrices**—powerful tools that represent data and transformations
in machine learning (ML). In this post, we’ll cover the mathematical foundations
of matrices, their role in ML as both data containers and transformation
operators, and how to work with them in Python using **NumPy** and **PyTorch**.
We’ll also visualize matrices and provide Python exercises to deepen your
understanding.

---

## The Math: Matrices as Data & Transformations

### What is a Matrix?

A **matrix** is a rectangular array of scalars arranged in rows and columns. A
matrix $ A $ with $ m $ rows and $ n $ columns (an $ m \times n $ matrix) is
written as:

$$
A = \begin{bmatrix} a*{11} & a*{12} & \cdots & a*{1n} \\ a*{21} & a*{22} &
\cdots & a*{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a*{m1} & a*{m2} & \cdots
& a\_{mn} \end{bmatrix}
$$

Each element $ a_{ij} $ is a scalar, where $ i $ is the row index and $ j $ is
the column index. In ML, matrices are used to:

- **Represent Data**: Datasets (e.g., rows as samples, columns as features) or
  images (e.g., pixel intensities).
- **Perform Transformations**: Linear layers in neural networks or geometric
  transformations like rotations.

### Matrices as Data

A matrix can represent structured data. For example:

- A **dataset matrix** might have rows as data points and columns as features. A
  dataset with 100 samples and 5 features is a $ 100 \times 5 $ matrix.
- An **image** (e.g., grayscale) is a matrix where each element is a pixel
  intensity. A 28x28 MNIST image is a $ 28 \times 28 $ matrix.

### Matrices as Transformations

A matrix can act as a **linear transformation**, mapping vectors to new vectors.
For an $ m \times n $ matrix $ A $ and a vector $ \mathbf{x} \in \mathbb{R}^n $,
the transformation is:

$$
\mathbf{y} = A \mathbf{x}
$$

where $ \mathbf{y} \in \mathbb{R}^m $. Each element of $ \mathbf{y} $ is a
linear combination of $ \mathbf{x} $’s components, weighted by the rows of $ A
$. In ML, this is the core of linear layers in neural networks, where $ A $
represents weights.

Geometrically, matrices can rotate, scale, or shear vectors. For example, a 2D
rotation matrix by angle $ \theta $:

$$
R = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta
\end{bmatrix}
$$

rotates a vector in $ \mathbb{R}^2 $ counterclockwise by $ \theta $.

---

## ML Context: Why Matrices Matter

Matrices are central to ML:

- **Data Representation**: Datasets are stored as matrices for efficient
  computation. Images, like those in computer vision, are matrices (or tensors
  for color images).
- **Linear Layers**: In neural networks, a layer’s computation is $ \mathbf{y} =
  W \mathbf{x} + \mathbf{b} $, where $ W $ is a weight matrix, $ \mathbf{x} $ is
  the input vector, and $ \mathbf{b} $ is a bias vector.
- **Transformations**: Matrices enable feature transformations, such as
  dimensionality reduction (e.g., PCA) or geometric operations in graphics and
  robotics.

Understanding matrices helps you manipulate data and design models efficiently.

---

## Python Code: Working with Matrices

Let’s explore matrices using **NumPy** and **PyTorch**, focusing on data
representation (e.g., images) and transformations (e.g., rotations). We’ll also
visualize matrices and their effects.

### Setup

Install the required libraries if needed:

```bash
pip install numpy torch matplotlib torchvision
```

### Matrices as Data: Loading an MNIST Image

Let’s load an MNIST image and treat it as a matrix:

```python
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# Get a single image
image, label = mnist_dataset[0]

# Convert to NumPy array (shape: 1, 28, 28 -> 28, 28)
image_matrix = image.squeeze().numpy()

# Print shape and sample elements
print("Image matrix shape:", image_matrix.shape)
print("Top-left 3x3 corner:\n", image_matrix[:3, :3])

# Visualize the image matrix
plt.figure(figsize=(5("Image as Matrix")
plt.imshow(image_matrix, cmap='gray')
plt.title(f"MNIST Digit: {label}")
plt.colorbar(label='Pixel Intensity')
plt.show()
```

**Output:**

```
Image matrix shape: (28, 28)
Top-left 3x3 corner:
 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]
```

This code loads an MNIST image as a $ 28 \times 28 $ matrix, prints its shape
and a 3x3 corner, and visualizes it as a grayscale image.

### Reshaping Matrices

Matrices can be reshaped to change their dimensions while preserving data. Let’s
flatten the image into a vector:

```python
# Flatten the image matrix into a vector
image_vector = image_matrix.flatten()

# Print new shape
print("Flattened vector shape:", image_vector.shape)

# Reshape back to 28x28
image_matrix_reshaped = image_vector.reshape(28, 28)

# Verify shapes match
print("Reshaped matrix shape:", image_matrix_reshaped.shape)
```

**Output:**

```
Flattened vector shape: (784,)
Reshaped matrix shape: (28, 28)
```

This demonstrates how images can be converted to vectors for ML models and
reshaped back.

### Matrices as Transformations: Rotation

Let’s apply a rotation matrix to a 2D vector:

```python
# Define a 2D vector
vector = np.array([1, 0])

# Define a 90-degree rotation matrix (pi/2 radians)
theta = np.pi / 2
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Apply transformation
rotated_vector = rotation_matrix @ vector  # Matrix-vector multiplication

# Print results
print("Original vector:", vector)
print("Rotation matrix:\n", rotation_matrix)
print("Rotated vector:", rotated_vector)

# Visualize original and rotated vectors
def plot_2d_vectors(vectors, labels, colors):
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
    plt.title("Vector Rotation")
    plt.show()

plot_2d_vectors(
    [vector, rotated_vector],
    ['Original', 'Rotated'],
    ['blue', 'red']
)
```

**Output:**

```
Original vector: [1 0]
Rotation matrix:
 [[ 6.123234e-17 -1.000000e+00]
  [ 1.000000e+00  6.123234e-17]]
Rotated vector: [ 6.123234e-17  1.000000e+00]
```

This rotates the vector $ [1, 0] $ by 90 degrees, resulting in $ [0, 1] $ (with
small numerical errors), and plots both vectors.

### PyTorch: Matrix Operations

Let’s perform the same rotation using PyTorch:

```python
# Convert to PyTorch tensors
vector_torch = torch.tensor([1.0, 0.0])
rotation_matrix_torch = torch.tensor([
    [torch.cos(torch.tensor(np.pi/2)), -torch.sin(torch.tensor(np.pi/2))],
    [torch.sin(torch.tensor(np.pi/2)), torch.cos(torch.tensor(np.pi/2))]
])

# Matrix-vector multiplication
rotated_vector_torch = rotation_matrix_torch @ vector_torch

print("PyTorch rotated vector:", rotated_vector_torch)
```

**Output:**

```
PyTorch rotated vector: tensor([ 0.,  1.])
```

This confirms PyTorch’s matrix operations align with NumPy’s.

---

## Exercises

Try these Python exercises to solidify your understanding. Solutions will be
discussed in the next post!

1. **Matrix Creation**: Create a $ 3 \times 4 $ matrix in NumPy filled with
   random integers between 0 and 9. Print the matrix and its transpose.
2. **Image Reshaping**: Load an MNIST image, reshape it into a $ 14 \times 56 $
   matrix, and visualize it. Compare it to the original $ 28 \times 28 $ image.
3. **Matrix Transformation**: Define a scaling matrix $ S = \begin{bmatrix} 2 &
   0 \\ 0 & 3 \end{bmatrix} $. Apply it to the vector $ [1, 1] $ using NumPy and
   plot the original and scaled vectors.
4. **PyTorch Matrix Multiplication**: Convert the scaling matrix and vector from
   Exercise 3 to PyTorch tensors, perform the multiplication, and verify the
   result matches NumPy’s.
5. **Dataset as Matrix**: Create a synthetic dataset with 5 samples and 3
   features (e.g., height, weight, age) as a $ 5 \times 3 $ NumPy matrix.
   Compute the mean of each feature (column).
6. **Rotation Animation**: Modify the rotation code to apply rotation matrices
   for angles $ \theta = 0^\circ, 45^\circ, 90^\circ $ to the vector $ [1, 0] $.
   Plot all resulting vectors in one 2D plot.

---

## What’s Next?

In the next post, we’ll explore **matrix arithmetic**—addition, scaling, and
multiplication—and their roles in ML, such as linear combinations and weighted
sums. We’ll dive deeper into NumPy and PyTorch operations with more examples and
exercises.

Happy learning, and see you in Part 3!
