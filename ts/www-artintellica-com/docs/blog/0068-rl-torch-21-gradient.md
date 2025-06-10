+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.1: Introduction to Gradient Descent—Math and Code"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Welcome to **Module 2: Optimization and Learning**! You’ve learned how to
represent and manipulate data in vector and matrix form. But how do we actually
**learn** from data? The answer is **optimization**—specifically, using
gradients to iteratively improve parameters by minimizing some loss function.
This is the backbone of nearly all deep learning and reinforcement learning
algorithms.

In this post, you’ll:

- Understand the mathematical basis of gradient descent, the core algorithm for
  machine learning optimization.
- Implement gradient descent for both scalar and vector functions in PyTorch.
- Visualize how optimization proceeds (and sometimes fails!) for different
  learning rates.
- Build crucial intuitions for the next steps: neural networks, policy
  gradients, Q-learning and beyond.

Let’s start learning—by making a machine learn!

---

## Mathematics: Gradient Descent

### What is Gradient Descent?

Gradient descent is a method for finding the local minimum of a function by
moving **against** the gradient (the direction of steepest ascent).

### Scalar Gradient Descent

Given a differentiable function $f(x)$, gradient descent updates the parameter
$x$ using:

$$
x \leftarrow x - \eta \, f'(x)
$$

where:

- $f'(x)$ is the derivative (gradient) at $x$,
- $\eta > 0$ is the **learning rate**.

For $f(x) = x^2$, $f'(x) = 2x$.

### Vector Gradient Descent

For a function $f(\mathbf{x})$ with $\mathbf{x} \in \mathbb{R}^n$:

$$
\mathbf{x} \leftarrow \mathbf{x} - \eta \, \nabla_{\mathbf{x}} f(\mathbf{x})
$$

where $\nabla_{\mathbf{x}} f(\mathbf{x})$ is the vector of partial derivatives
(gradient).

For $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \sum_i x_i^2$,
$\nabla_{\mathbf{x}} f(\mathbf{x}) = 2\mathbf{x}$.

---

## Python Demonstrations

### Demo 1: Scalar Gradient Descent for $f(x) = x^2$

Let’s do it by hand (without autograd for now).

```python
def grad_fx(x: float) -> float:
    # Derivative of f(x) = x^2 is 2x
    return 2 * x

x = 5.0  # Start far from zero
eta = 0.1  # Learning rate

trajectory = [x]
for step in range(20):
    x = x - eta * grad_fx(x)
    trajectory.append(x)
print("Final x:", x)
```

---

### Demo 2: Visualize the Optimization Path

```python
import numpy as np
import matplotlib.pyplot as plt

# The function and its minimum
def fx(x):
    return x**2

# Use trajectory from previous demo
steps = np.array(trajectory)
plt.plot(steps, fx(steps), 'o-', label="Optimization Path")
plt.plot(0, 0, 'rx', markersize=12, label="Minimum")
plt.xlabel('x value')
plt.ylabel('f(x)')
plt.title('Gradient Descent for $f(x) = x^2$')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Demo 3: Vector Gradient Descent for $f(\mathbf{x}) = ||\mathbf{x}||^2$

```python
import torch

def grad_f_vec(x: torch.Tensor) -> torch.Tensor:
    return 2 * x

x: torch.Tensor = torch.tensor([5.0, -3.0], dtype=torch.float32)  # Initial point in 2D
eta_vec = 0.2
trajectory_vec = [x.clone()]

for step in range(15):
    x = x - eta_vec * grad_f_vec(x)
    trajectory_vec.append(x.clone())

trajectory_vec = torch.stack(trajectory_vec)

print("Final x:", x)
print("Norm at end:", torch.norm(x).item())
```

---

### Demo 4: Learning Rate Experiment

Let’s try different learning rates and see their effect.

```python
init_x = 5.0
learning_rates = [0.05, 0.2, 0.8, 1.01]
colors = ['b', 'g', 'r', 'orange']

plt.figure()
for lr, col in zip(learning_rates, colors):
    x = init_x
    hist = [x]
    for _ in range(12):
        x = x - lr * grad_fx(x)
        hist.append(x)
    plt.plot(hist, fx(np.array(hist)), 'o-', color=col, label=f'LR={lr}')
plt.plot(0, 0, 'kx', markersize=12)
plt.title('Gradient Descent Paths for Different Learning Rates')
plt.xlabel('x value')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

- **Small $\eta$**: slow convergence.
- **Large $\eta$**: may overshoot or diverge.

---

## Exercises

Put your new optimization skills to the test:

### **Exercise 1:** Implement Scalar Gradient Descent for $f(x) = (x-3)^2$

- Write a function that starts from $x_0 = -7$ and uses gradient descent for 20
  steps.
- Print $x$ after each step.

### **Exercise 2:** Visualize the Optimization Path on a 2D Plot

- Plot $f(x)$ and overlay the path of $x$ values as you optimize.

### **Exercise 3:** Use Vector Gradient Descent on $f(\mathbf{x}) = ||\mathbf{x} - [2, -1]||^2$

- Start from $\mathbf{x}_0 = [5, 5]$.
- Use 20 steps and plot the path in 2D.

### **Exercise 4:** Experiment with Different Learning Rates and Observe Convergence

- Try $\eta = 0.01$, $0.1$, $1.0$, $1.5$ for both scalar and vector cases.
- Plot and compare trajectories.

---

### **Sample Starter Code for Exercises**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 1
def grad_fx_shifted(x: float) -> float:
    return 2 * (x - 3)

x = -7.0
eta = 0.2
traj = [x]
for _ in range(20):
    x = x - eta * grad_fx_shifted(x)
    traj.append(x)
    print(f"x = {x:.4f}")

# EXERCISE 2
x_arr = np.array(traj)
plt.plot(x_arr, (x_arr - 3)**2, 'o-')
plt.plot(3, 0, 'rx', label='Minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Optimization Path: Scalar')
plt.legend(); plt.grid(True)
plt.show()

# EXERCISE 3
def grad_fvec_shifted(x: torch.Tensor) -> torch.Tensor:
    return 2 * (x - torch.tensor([2.0, -1.0]))

xv = torch.tensor([5.0, 5.0])
eta_vec = 0.1
traj_v = [xv.clone()]
for _ in range(20):
    xv = xv - eta_vec * grad_fvec_shifted(xv)
    traj_v.append(xv.clone())

traj_v_np = torch.stack(traj_v).numpy()
target = np.array([2.0, -1.0])
plt.plot(traj_v_np[:,0], traj_v_np[:,1], 'o-', label='GD Path')
plt.plot(target[0], target[1], 'rx', label='Minimum')
plt.title('Gradient Descent Path: Vector')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True)
plt.show()

# EXERCISE 4
lrs = [0.01, 0.1, 1.0, 1.5]
plt.figure()
for lr in lrs:
    x = -7.0
    h = [x]
    for _ in range(15):
        x = x - lr * grad_fx_shifted(x)
        h.append(x)
    plt.plot(h, [(hx-3)**2 for hx in h], 'o-', label=f'LR={lr}')
plt.plot(3, 0, 'kx', markersize=10, label='Minimum')
plt.legend(); plt.grid(True)
plt.title('Learning Rate Effect: Scalar')
plt.xlabel('x'); plt.ylabel('f(x)')
plt.show()
```

---

## Conclusion

You’ve stepped into the engine room of all learning systems: **optimization via
gradient descent**.

- You coded scalar and vector gradient descent by hand.
- You visualized the entire learning process and saw the powerful (and sometimes
  chaotic) effect of learning rates.
- You’ve laid the mathematical and conceptual foundation for everything from
  simple regressions to deep RL.

**Next:** We’ll dive into automatic differentiation—how PyTorch “automagically”
computes gradients for any function you can imagine. This will let you optimize
neural networks, RL objectives, and more.

_Keep experimenting with functions, rates, and dimensions; mastery of
optimization is the key to all modern AI. See you in Part 2.2!_
