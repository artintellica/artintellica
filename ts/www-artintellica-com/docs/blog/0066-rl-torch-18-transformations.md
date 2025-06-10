+++
title = "Learn Reinforcement Learning with PyTorch, Part 1.8: Linear Transformations and Simple Data Transformations"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0066-rl-torch-18-transformations"
+++

## Introduction

Welcome to Part 1.8 of Artintellica’s open-source RL with PyTorch course! So
far, you’ve mastered the geometry of tensors—norms, distances, and projections.
Now, you’ll see how matrices actually **transform** data. This is the essence of
linear algebra in ML: weights in neural networks, value function features, even
environment state transitions are, at their core, linear transformations.

In this post, you’ll:

- Create and understand rotation and scaling matrices.
- Apply them to data points and see the effect geometrically.
- Visualize how transformations morph shapes (like rotating a square!).
- Chain multiple transformations for composite effects—a crucial technique in
  data preprocessing, augmentation, and deep learning layers.

Let’s turn math into moving pictures!

---

## Mathematics: Linear Transformations

A **linear transformation** can be represented by a matrix $A$. Given a vector
$\mathbf{x}$, the transformation maps it to $\mathbf{y}$:

$$
\mathbf{y} = A \mathbf{x}
$$

The most common 2D transformations are:

### Rotation

To rotate a vector by angle $\theta$ counterclockwise about the origin:

$$
R(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

### Scaling

To scale by $s_x$ in $x$ direction and $s_y$ in $y$:

$$
S(s_x, s_y) = \begin{bmatrix}
s_x & 0 \\
0 & s_y
\end{bmatrix}
$$

### Chaining Transformations

Chaining means applying transformations with $A_2A_1\mathbf{x}$ (matrix
multiplication order: **right-to-left**).

---

## Python Demonstrations

Let’s implement and visualize these concepts in PyTorch.

### Demo 1: Create Rotation and Scaling Matrices

```python
import torch
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

theta = math.pi / 4  # 45 degrees
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
print("Rotation (45°):\n", R)
print("Scaling (2, 0.5):\n", S)
```

---

### Demo 2: Apply a Rotation Matrix to a Set of 2D Points

Let’s rotate some points.

```python
# Array of 2D points (shape Nx2)
points: torch.Tensor = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0]
])
theta = math.pi / 2   # 90 degrees
R = rotation_matrix(theta)  # Counterclockwise
rotated: torch.Tensor = points @ R.T
print("Original points:\n", points)
print("Rotated points (90°):\n", rotated)
```

_Note:_ We use `@ R.T` because points are (N,2) \* (2,2) → (N,2).

---

### Demo 3: Visualize the Effect of a Transformation on a Shape (a Square)

Let’s rotate and scale a square, and plot before and after.

```python
import matplotlib.pyplot as plt

# Define points: corners of the square (counterclockwise)
square: torch.Tensor = torch.tensor([
    [1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0]    # close the square for the plot
])
# Transformation: rotate by 30° and scale x2 in x, 0.5 in y
theta = math.radians(30)
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
# Apply scaling THEN rotation
transformed: torch.Tensor = (square @ S.T) @ R.T

plt.figure(figsize=(6,6))
plt.plot(square[:,0], square[:,1], 'bo-', label='Original')
plt.plot(transformed[:,0], transformed[:,1], 'ro-', label='Transformed')
plt.title("Transforming a Square: Rotation and Scaling")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
```

---

### Demo 4: Chain Multiple Transformations and Describe the Result

Let’s build the matrix for scaling-then-rotation, versus rotation-then-scaling
(order matters!).

```python
# Chain: first scaling, then rotation (note order!)
composite1 = R @ S
# Chain: first rotation, then scaling
composite2 = S @ R

point = torch.tensor([[1.0, 0.0]])
result1 = (point @ composite1.T).squeeze()
result2 = (point @ composite2.T).squeeze()
print("Scaling THEN rotation:", result1)
print("Rotation THEN scaling:", result2)
```

**Describe the result:** The order of operations changes the outcome! Try
swapping R and S in the square plot as an experiment.

---

## Exercises

Try these in your own script or notebook!

### **Exercise 1:** Create Rotation and Scaling Matrices

- Create a 2D rotation matrix for 60 degrees.
- Create a scaling matrix that doubles $x$ and halves $y$.
- Print both matrices.

### **Exercise 2:** Apply a Rotation Matrix to a Set of 2D Points

- Define three points: $(1,0)$, $(0,1)$, $(1,1)$.
- Rotate them by 45 degrees using your rotation matrix.
- Print the original and rotated coordinates.

### **Exercise 3:** Visualize the Effect of a Transformation on a Shape (e.g., a Square)

- Define coordinates for the corners of a square.
- Apply a scaling transformation with $s_x = 1.5$, $s_y = 0.5$.
- Plot the original and scaled squares.

### **Exercise 4:** Chain Multiple Transformations and Describe the Result

- Use your square from Exercise 3.
- Apply first a rotation by 90 degrees, then a scale of $0.5$ in $x$ and $2$ in
  $y$ (and plot the result).
- Then swap the order: scale first, then rotate.
- Visualize and describe the geometric difference you see.

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
theta = math.radians(60)
R = rotation_matrix(theta)
S = scaling_matrix(2.0, 0.5)
print("Rotation 60°:\n", R)
print("Scaling x2, y0.5:\n", S)

# EXERCISE 2
pts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
rot45 = rotation_matrix(math.radians(45))
rotated = pts @ rot45.T
print("Original points:\n", pts)
print("Rotated by 45°:\n", rotated)

# EXERCISE 3
square = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]], dtype=torch.float32)
S2 = scaling_matrix(1.5, 0.5)
scaled_sq = square @ S2.T
plt.figure(figsize=(5,5))
plt.plot(square[:,0], square[:,1], 'b-o', label='Original Square')
plt.plot(scaled_sq[:,0], scaled_sq[:,1], 'r-o', label='Scaled Square')
plt.title("Scaling a Square")
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal'); plt.grid(True); plt.legend(); plt.show()

# EXERCISE 4
R90 = rotation_matrix(math.radians(90))
S3 = scaling_matrix(0.5, 2.0)
# rotate then scale
sq_rot_scale = (square @ R90.T) @ S3.T
# scale then rotate
sq_scale_rot = (square @ S3.T) @ R90.T

plt.figure(figsize=(6,6))
plt.plot(square[:,0], square[:,1], 'b--o', label='Original')
plt.plot(sq_rot_scale[:,0], sq_rot_scale[:,1], 'r-o', label='Rotate->Scale')
plt.plot(sq_scale_rot[:,0], sq_scale_rot[:,1], 'g-o', label='Scale->Rotate')
plt.title("Chained Transformations on a Square")
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal'); plt.grid(True); plt.legend(); plt.show()

# Describe your observations!
```

---

## Conclusion

Today, you visualized and coded the _true power_ of matrices in ML: they
transform data! You learned how to:

- Build and apply rotation/scaling matrices;
- Transform point clouds and shapes;
- Chain transformations and observe the order’s critical impact.

**Next:** We’ll use your knowledge to load, manipulate, and visualize
real/simulated data using the full PyTorch matrix toolbox. You'll be ready for
data prep, debugging, and even RL environments!

_Practice these shape-changing operations—they'll be everywhere in RL and deep
learning! See you in Part 1.9._
