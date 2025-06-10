+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.5: Fitting Nonlinear Curves—Polynomial Regression"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0072-rl-torch-25-nonlinear"
+++

## Introduction

Welcome back! After mastering fitting straight lines, let's tackle the obstacle
every real-world modeler faces: **nonlinear patterns.** Most data in
reinforcement learning—and in nature—cannot be captured by a single line. The
next step is to fit curves, not just lines, using **polynomial regression**.

In this post, you'll:

- Generate and visualize synthetic nonlinear (polynomial) data.
- Construct polynomial features explicitly in PyTorch.
- Fit both linear and polynomial regressions and compare their strengths.
- Visualize **overfitting** as polynomial degree increases—a fundamental concept
  in machine learning!

---

## Mathematics: Polynomial Regression

In **polynomial regression**, the model predicts $y$ using a polynomial in $x$:

$$
y \approx w_0 + w_1 x + w_2 x^2 + \ldots + w_d x^d
$$

Or in vector/tensor notation:

$$
y \approx \mathbf{w}^T \mathbf{\phi}(x)
$$

where $\mathbf{\phi}(x) = [1, x, x^2, ..., x^d]$ is the **feature vector**.

We fit the weights $\mathbf{w}$ by minimizing mean squared error (as in linear
regression, but now with more complex features).

---

## Python Demonstrations

### Demo 1: Generate Polynomial Data with Noise

```python
import torch
import matplotlib.pyplot as plt

N = 120
torch.manual_seed(0)
x = torch.linspace(-3, 3, N)
# True relationship: cubic (degree 3) with noise
y_true = 0.4 * x**3 - x**2 + 0.5 * x + 2.0
y = y_true + 2.0 * torch.randn(N)

plt.scatter(x, y, alpha=0.5, label='Data (noisy)')
plt.plot(x, y_true, 'k--', label='True curve')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Synthetic Polynomial Data")
plt.legend(); plt.show()
```

---

### Demo 2: Fit a Polynomial Using Explicit Feature Construction

```python
# Helper: build polynomial feature matrix (design matrix) Phi(x), for degree d
def poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    return torch.stack([x**i for i in range(degree+1)], dim=1)  # Shape: (N, degree+1)

degree = 3  # Try cubic first
X_poly = poly_features(x, degree)  # (N, 4)
w = torch.zeros(degree+1, requires_grad=True)
lr = 0.0002

losses = []
for epoch in range(3000):
    y_pred = X_poly @ w
    loss = ((y_pred - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    losses.append(loss.item())
    if epoch % 500 == 0 or epoch == 2999:
        print(f"Epoch {epoch}: loss={loss.item():.3f}")

plt.plot(losses)
plt.title("Training Loss (Polynomial Regression)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
```

---

### Demo 3: Compare Performance of Linear vs. Polynomial Regression

Fit and plot both a line and a cubic polynomial.

```python
# Linear
X_lin = poly_features(x, 1)
w_lin = torch.zeros(2, requires_grad=True)
for epoch in range(400):
    y_pred_lin = X_lin @ w_lin
    loss_lin = ((y_pred_lin - y) ** 2).mean()
    loss_lin.backward()
    with torch.no_grad():
        w_lin -= 0.001 * w_lin.grad
    w_lin.grad.zero_()

# Cubic (reuse from above)
with torch.no_grad():
    y_fit_lin = X_lin @ w_lin
    y_fit_poly = X_poly @ w

plt.scatter(x, y, alpha=0.3, label="Noisy data")
plt.plot(x, y_true, 'k--', label="True function")
plt.plot(x, y_fit_lin, 'b-', label="Linear fit")
plt.plot(x, y_fit_poly, 'r-', label="Polynomial fit")
plt.legend(); plt.xlabel("x"); plt.ylabel("y")
plt.title("Linear vs. Polynomial Regression")
plt.show()
```

---

### Demo 4: Visualize Overfitting by Increasing the Polynomial Degree

Vary $d$ and plot fits.

```python
degrees = [1, 3, 8, 15]
plt.scatter(x, y, alpha=0.2, label="Noisy Data")

colors = ["b", "g", "orange", "r"]
for deg, c in zip(degrees, colors):
    w = torch.zeros(deg+1, requires_grad=True)
    Xp = poly_features(x, deg)
    for epoch in range(800):
        y_pred = Xp @ w
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= 0.001 * w.grad
        w.grad.zero_()
    with torch.no_grad():
        y_fit = Xp @ w
    plt.plot(x, y_fit, color=c, label=f"degree={deg}")

plt.plot(x, y_true, "k--", label="True curve")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.title("Polynomial Degree and Overfitting")
plt.show()
```

Notice how higher degrees can "curve" better to the data --- but can also wildly
overfit noise.

---

## Exercises

### **Exercise 1:** Generate Polynomial Data with Noise

- True function: $y_{\text{true}} = -0.5x^3 + 1.2x^2 - 0.7x + 4$.
- Generate $N=150$ points in $x$ from $-2.5$ to $2.5$.
- $y = y_{\text{true}} + \text{Gaussian noise (std=1.2)}$.
- Plot $x$ and $y$ with the true curve.

---

### **Exercise 2:** Fit a Polynomial Using Explicit Feature Construction in PyTorch

- Construct the feature matrix up to degree $d=3$.
- Initialize weights as zeros with `requires_grad=True`.
- Train using a loop as above with MSE loss.
- Plot the loss over epochs.

---

### **Exercise 3:** Compare Performance of Linear vs. Polynomial Regression

- Fit both a linear ($d=1$) and a cubic ($d=3$) regression to your data.
- Plot their fits together with the real data and the "true" curve.

---

### **Exercise 4:** Visualize Overfitting by Increasing Polynomial Degree

- Fit polynomials with degrees: 1, 3, 7, and 15.
- Plot all fits. What happens to degree 15? Overfitting!

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

def poly_features(x: torch.Tensor, d: int) -> torch.Tensor:
    return torch.stack([x**i for i in range(d+1)], dim=1)

# EXERCISE 1
N = 150
x = torch.linspace(-2.5, 2.5, N)
y_true = -0.5 * x**3 + 1.2 * x**2 - 0.7 * x + 4
torch.manual_seed(0)
y = y_true + 1.2 * torch.randn(N)
plt.scatter(x, y, alpha=0.5)
plt.plot(x, y_true, 'k--', label='True function')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Cubic Data'); plt.show()

# EXERCISE 2
degree = 3
Xp = poly_features(x, degree)
w = torch.zeros(degree+1, requires_grad=True)
losses = []
for epoch in range(2500):
    y_pred = Xp @ w
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.001 * w.grad
    w.grad.zero_()
    losses.append(loss.item())
plt.plot(losses); plt.title('Polynomial Regression Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()

# EXERCISE 3
# Linear fit
Xl = poly_features(x, 1)
wl = torch.zeros(2, requires_grad=True)
for epoch in range(800):
    y_pred_lin = Xl @ wl
    loss_lin = ((y_pred_lin - y) ** 2).mean()
    loss_lin.backward()
    with torch.no_grad():
        wl -= 0.001 * wl.grad
    wl.grad.zero_()
with torch.no_grad():
    y_fit_lin = Xl @ wl
    y_fit_poly = Xp @ w
plt.scatter(x, y, alpha=0.3)
plt.plot(x, y_true, "k--", label="True function")
plt.plot(x, y_fit_lin, "b-", label="Linear fit")
plt.plot(x, y_fit_poly, "r-", label="Cubic fit")
plt.legend(); plt.show()

# EXERCISE 4
degrees = [1, 3, 7, 15]
colors = ['b', 'g', 'orange', 'r']
plt.scatter(x, y, alpha=0.2, label="Noisy data")
for deg, col in zip(degrees, colors):
    w = torch.zeros(deg + 1, requires_grad=True)
    Xd = poly_features(x, deg)
    for epoch in range(1000):
        y_pred = Xd @ w
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= 0.0005 * w.grad
        w.grad.zero_()
    with torch.no_grad():
        y_fit = Xd @ w
    plt.plot(x, y_fit, color=col, label=f"d={deg}")
plt.plot(x, y_true, "k--", label="True")
plt.legend(); plt.title("Overfitting Example"); plt.show()
```

---

## Conclusion

You now know how to:

- Generate and fit models to nonlinear data with PyTorch.
- Explicitly build and learn polynomial features.
- Identify and _visualize_ overfitting, the eternal nemesis of powerful models.

**Next:** We step into the world of **classification**: you’ll fit linear models
for _categories_ instead of _regression_—a turning point toward deep RL and
real-world applications.

_Experiment with different polynomial degrees and noise levels—these are the
foundations for model selection and understanding generalization in RL and
beyond. See you in Part 2.6!_
