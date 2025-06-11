+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.6: Classification Basics—Logistic Regression"
author = "Artintellica"
date = "2024-06-10"
+++

## Introduction

Linear regression is great for continuous outcomes, but most RL problems involve **making decisions**—classifying between actions, states, or outcomes. The simplest classification model is **logistic regression**, which uses a “soft” step function to produce probabilities and binary classifications from input features.

In this post you will:

- Generate and visualize synthetic binary classification data.
- Implement logistic regression “from scratch” using tensors (sigmoid and binary cross-entropy).
- Train with PyTorch’s optimizer and compare both approaches.
- Plot the decision boundary and evaluate accuracy.

Let’s open the door to classification—the bedrock of RL decision-making.

---

## Mathematics: Logistic Regression

Given input vector $\mathbf{x} \in \mathbb{R}^d$, **logistic regression** predicts:

$$
y = \begin{cases}
1 & \text{if } \sigma(\mathbf{w}^T\mathbf{x} + b) > 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the **sigmoid** function (maps real numbers to $(0, 1)$ probability).
- Model parameters: $\mathbf{w}$ (weights), $b$ (bias).

The **binary cross-entropy (BCE) loss** for a dataset $(\mathbf{x}_i, y_i)$ is:
$$
L_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i\log p_i + (1-y_i)\log(1-p_i)\right]
$$
where $p_i = \sigma(\mathbf{w}^\top\mathbf{x}_i + b)$.

---

## Python Demonstrations

### Demo 1: Generate Binary Classification Data

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 200
# Two Gaussian blobs
mean0 = torch.tensor([-2.0, 0.0])
mean1 = torch.tensor([2.0, 0.5])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.2]])
L = torch.linalg.cholesky(cov)

X0 = torch.randn(N//2, 2) @ L.T + mean0
X1 = torch.randn(N//2, 2) @ L.T + mean1
X = torch.cat([X0, X1], dim=0)
y = torch.cat([torch.zeros(N//2), torch.ones(N//2)])

plt.scatter(X0[:,0], X0[:,1], color='b', alpha=0.5, label="Class 0")
plt.scatter(X1[:,0], X1[:,1], color='r', alpha=0.5, label="Class 1")
plt.xlabel('x1'); plt.ylabel('x2')
plt.legend(); plt.title("Binary Classification Data")
plt.show()
```

---

### Demo 2: Logistic Regression “From Scratch” (Sigmoid + BCE)

```python
def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))

# Add bias feature for simplicity: [x1, x2, 1]
X_aug = torch.cat([X, torch.ones(N,1)], dim=1)  # shape (N, 3)
w = torch.zeros(3, requires_grad=True)
lr = 0.05

losses = []
for epoch in range(1500):
    z = X_aug @ w              # Linear
    p = sigmoid(z)             # Probabilities
    # Numerical stabilization: clamp p
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    bce = (-y * torch.log(p) - (1 - y) * torch.log(1 - p)).mean()
    bce.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    losses.append(bce.item())
    if epoch % 300 == 0 or epoch == 1499:
        print(f"Epoch {epoch}: BCE loss={bce.item():.3f}")

plt.plot(losses)
plt.title("Training Loss: Logistic Regression (Scratch)")
plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.grid(True)
plt.show()
```

---

### Demo 3: Train with PyTorch Optimizer and Compare

```python
w2 = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.SGD([w2], lr=0.05)
losses2 = []
for epoch in range(1500):
    z = X_aug @ w2
    p = sigmoid(z)
    p = p.clamp(1e-8, 1 - 1e-8)
    bce = torch.nn.functional.binary_cross_entropy(p, y)
    bce.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(bce.item())

plt.plot(losses2, label='Optimizer loss')
plt.title("PyTorch Optimizer BCE Loss")
plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.grid(True)
plt.show()

print("Final weights (manual):", w.detach().numpy())
print("Final weights (optimizer):", w2.detach().numpy())
```

---

### Demo 4: Plot Decision Boundary and Accuracy

```python
import numpy as np

with torch.no_grad():
    # Grid for decision boundary
    x1g, x2g = torch.meshgrid(torch.linspace(-6, 6, 100), torch.linspace(-4, 5, 100), indexing='ij')
    Xg = torch.stack([x1g.reshape(-1), x2g.reshape(-1), torch.ones(100*100)], dim=1)
    p_grid = sigmoid(Xg @ w2).reshape(100, 100)
    
    # Predictions for accuracy
    preds = (sigmoid(X_aug @ w2) > 0.5).float()
    acc = (preds == y).float().mean().item()

plt.contourf(x1g, x2g, p_grid, levels=[0,0.5,1], colors=['lightblue','salmon'], alpha=0.2)
plt.scatter(X0[:,0], X0[:,1], color='b', alpha=0.5, label="Class 0")
plt.scatter(X1[:,0], X1[:,1], color='r', alpha=0.5, label="Class 1")
plt.title(f"Decision Boundary (Accuracy: {acc*100:.1f}%)")
plt.xlabel('x1'); plt.ylabel('x2')
plt.legend(); plt.show()
```

---

## Exercises

### **Exercise 1:** Generate Binary Classification Data

- Make two clouds in 2D using Gaussians with means $[0, 0]$ and $[3, 2]$ and a shared covariance.
- Stack them to form $N=100$ dataset and make integer labels $0$ and $1$.
- Plot the dataset, color by class.

---

### **Exercise 2:** Implement Logistic Regression “From Scratch” (Sigmoid + BCE)

- Add a bias column to data.
- Initialize weights as zeros (with `requires_grad=True`).
- Use the sigmoid and BCE formulas explicitly in your training loop.
- Train for $1000$ epochs, plot the loss curve.

---

### **Exercise 3:** Train with PyTorch’s Optimizer and Compare Results

- Re-run the same setup, but use a PyTorch optimizer (`SGD` or `Adam`).
- Compare learned weights and loss curves.

---

### **Exercise 4:** Plot Decision Boundary and Accuracy

- Use the learned weights to plot the decision boundary over your scatterplot.
- Compute accuracy on all points.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# EXERCISE 1
torch.manual_seed(0)
N = 100
mean0, mean1 = torch.tensor([0., 0.]), torch.tensor([3., 2.])
cov = torch.tensor([[1.5, 0.3],[0.3, 1.0]])
L = torch.linalg.cholesky(cov)
X0 = torch.randn(N//2, 2) @ L.T + mean0
X1 = torch.randn(N//2, 2) @ L.T + mean1
X = torch.cat([X0, X1], dim=0)
y = torch.cat([torch.zeros(N//2), torch.ones(N//2)])
plt.scatter(X0[:,0], X0[:,1], c='blue', label="Class 0")
plt.scatter(X1[:,0], X1[:,1], c='red', label="Class 1")
plt.xlabel('x1'); plt.ylabel('x2'); plt.title("Binary Data"); plt.legend(); plt.show()

# EXERCISE 2
def sigmoid(z): return 1/(1 + torch.exp(-z))
X_aug = torch.cat([X, torch.ones(N,1)], dim=1)
w = torch.zeros(3, requires_grad=True)
lr = 0.06
losses = []
for epoch in range(1000):
    z = X_aug @ w
    p = sigmoid(z).clamp(1e-8, 1-1e-8)
    bce = (-y * torch.log(p) - (1-y)*torch.log(1-p)).mean()
    bce.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    losses.append(bce.item())
plt.plot(losses); plt.title("Logistic Regression Loss (Scratch)"); plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.show()

# EXERCISE 3
w2 = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.SGD([w2], lr=lr)
losses2 = []
for epoch in range(1000):
    z = X_aug @ w2
    p = sigmoid(z).clamp(1e-8, 1-1e-8)
    bce = torch.nn.functional.binary_cross_entropy(p, y)
    bce.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(bce.item())
plt.plot(losses2); plt.title("Logistic Regression Loss (Optim)"); plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.show()
print("Final weights (scratch):", w.data)
print("Final weights (optim):", w2.data)

# EXERCISE 4
with torch.no_grad():
    grid_x, grid_y = torch.meshgrid(torch.linspace(-3,6,120), torch.linspace(-2,5,120), indexing='ij')
    Xg = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(grid_x.numel())], dim=1)
    p_grid = sigmoid(Xg @ w2).reshape(120,120)
    preds = (sigmoid(X_aug @ w2) > 0.5).float()
    acc = (preds == y).float().mean().item()
plt.contourf(grid_x, grid_y, p_grid, levels=[0,0.5,1], colors=['lightblue','salmon'], alpha=0.23)
plt.scatter(X0[:,0], X0[:,1], c='b', label='Class 0', alpha=0.6)
plt.scatter(X1[:,0], X1[:,1], c='r', label='Class 1', alpha=0.6)
plt.title(f"Decision Boundary (Acc: {acc*100:.1f}%)")
plt.xlabel('x1'); plt.ylabel('x2'); plt.legend(); plt.show()
```

---

## Conclusion

You’ve learned to:

- Generate and visualize binary data.
- Code and optimize logistic regression with explicit tensor ops and with PyTorch’s optimizer.
- See the effect of decision boundaries and evaluate classification accuracy.

**Next up:** Multiclass classification—move beyond binary to solving problems where you need to pick from more than two possible actions or classes. We're almost ready for neural nets!

*Keep experimenting—classification is the true foundation of RL actions and decisions! See you in Part 2.7!*
