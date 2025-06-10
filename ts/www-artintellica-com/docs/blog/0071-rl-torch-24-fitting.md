+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.4: Fitting a Line—Linear Regression from Scratch with PyTorch"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0071-rl-torch-24-fitting"
+++

## Introduction

Welcome back! After exploring loss functions and cost surfaces, you’re ready for
your first “full stack” learning algorithm: **linear regression**. This is where
we connect optimization, loss, and data—finding the best-fit line for points by
minimizing error. Many deep RL algorithms ultimately rely on these same
principles!

In this post you’ll:

- Generate and visualize noisy, synthetic linear data.
- Implement linear regression _from scratch_ with PyTorch tensors and gradients.
- Use PyTorch’s optimizer to automate learning.
- Visualize model predictions versus real data, and compute the $R^2$
  goodness-of-fit score.

Let’s fit a line and watch your first model learn!

---

## Mathematics: Linear Regression and Least Squares

In classic **linear regression**, we assume $y \approx wx + b$, where $w$ is the
slope (weight) and $b$ is the intercept (bias).

The goal:

> Find $w, b$ that minimize the **mean squared error** loss:

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^n (wx_i + b - y_i)^2
$$

We optimize $w, b$ by computing their gradients (via autograd) and updating via
gradient descent.

---

## Python Demonstrations

### Demo 1: Generate Synthetic Linear Data with Noise

```python
import torch
import matplotlib.pyplot as plt

# True parameters
w_true = 2.5
b_true = -1.7
N = 120
torch.manual_seed(42)

# Generate random x and noisy y
x = torch.linspace(-3, 3, N)
y = w_true * x + b_true + 0.9 * torch.randn(N)

plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, w_true * x + b_true, 'k--', label='True line')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.title("Synthetic Linear Data"); plt.grid(True)
plt.show()
```

---

### Demo 2: Implement Linear Regression Training Loop from Scratch (Tensors Only)

```python
# Initialize parameters
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.04

losses = []
for epoch in range(80):
    y_pred = w * x + b              # Linear model
    loss = ((y_pred - y)**2).mean() # MSE
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == 79:
        print(f"Epoch {epoch:2d}: w={w.item():.2f}, b={b.item():.2f}, loss={loss.item():.3f}")

plt.plot(losses)
plt.title("Training Loss (From Scratch)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.grid(True)
plt.show()
```

---

### Demo 3: Use PyTorch’s Autograd and Optimizer to Fit a Line

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.04)

losses2 = []
for epoch in range(80):
    y_pred = w * x + b
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(loss.item())

print(f"Learned parameters: w={w.item():.2f}, b={b.item():.2f}")
plt.plot(losses2)
plt.title("Training Loss (With PyTorch Optimizer)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.grid(True)
plt.show()
```

---

### Demo 4: Plot Predictions vs. Ground Truth and Compute $R^2$ Score

```python
# Use learned w, b (could be from previous cell)
with torch.no_grad():
    y_fit = w * x + b

plt.scatter(x, y, label='Data', alpha=0.6)
plt.plot(x, w_true * x + b_true, 'k--', label='True line')
plt.plot(x, y_fit, 'r-', label='Fitted line')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title("Model Fit vs Ground Truth"); plt.grid(True)
plt.show()

# Compute R^2
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return 1 - ss_res / ss_tot

print("R^2 score:", r2_score(y, y_fit).item())
```

---

## Exercises

### **Exercise 1:** Generate Synthetic Linear Data with Noise

- True line: $w_\text{true} = 1.7$, $b_\text{true} = -0.3$.
- Sample $N=100$ points for $x$ from -2 to 4.
- $y = w_{\text{true}}x + b_{\text{true}} +$ noise (Gaussian, std=0.5).
- Plot $x$ and $y$ with the true line.

---

### **Exercise 2:** Implement Linear Regression Training Loop from Scratch (Only Tensors!)

- Randomly initialize $w, b$ (set `requires_grad=True`).
- For 100 epochs: predict, compute loss (MSE), backward, update $w, b$ with a
  learning rate (no optimizer object).
- Zero grads after each update.
- Plot the loss curve.

---

### **Exercise 3:** Use PyTorch’s Autograd and Optimizer to Fit a Line

- Use `torch.optim.SGD` or `torch.optim.Adam`.
- Train for 100 epochs.
- Plot loss vs. epoch and print the learned $w, b$.

---

### **Exercise 4:** Plot Predictions vs. Ground Truth and Compute $R^2$ Score

- Plot the original data, the true line, and the model’s predictions.
- Compute and print the $R^2$ score for your fitted model.

---

### **Sample Starter Code for Exercises**

```python
import torch
import matplotlib.pyplot as plt

# EXERCISE 1
w_true, b_true = 1.7, -0.3
N = 100
x = torch.linspace(-2, 4, N)
torch.manual_seed(0)
y = w_true * x + b_true + 0.5 * torch.randn(N)
plt.scatter(x, y, s=12, alpha=0.7)
plt.plot(x, w_true * x + b_true, 'k--', label='True line')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.title("Synthetic Data"); plt.show()

# EXERCISE 2
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.05
losses = []
for epoch in range(100):
    y_pred = w * x + b
    loss = ((y - y_pred)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()
    losses.append(loss.item())
plt.plot(losses)
plt.title("Loss Over Epochs (Scratch)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

# EXERCISE 3
w2 = torch.randn(1, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)
optimizer = torch.optim.SGD([w2, b2], lr=0.05)
losses2 = []
for epoch in range(100):
    y_pred = w2 * x + b2
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(loss.item())
print(f"Learned w={w2.item():.3f}, b={b2.item():.3f}")
plt.plot(losses2)
plt.title("Loss Over Epochs (Optimizer)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

# EXERCISE 4
with torch.no_grad():
    y_fit = w2 * x + b2
plt.scatter(x, y, alpha=0.6, label='Actual')
plt.plot(x, w_true * x + b_true, 'k--', label='True')
plt.plot(x, y_fit, 'r-', label='Predicted')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title("Prediction vs Ground Truth"); plt.show()
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return 1 - ss_res/ss_tot
print("R^2 score:", r2_score(y, y_fit).item())
```

---

## Conclusion

You’ve just fit your first model—a line—to data using PyTorch. You now know how
to:

- Simulate data, code a training loop, and use autograd to learn.
- Visualize loss, predictions, and compare with ground truth.
- Quantify the fit with $R^2$, a vital real-world skill for model evaluation.

**Up next:** Polynomial and nonlinear regression—seeing when and how linear
models break down, and how to model more complex data in PyTorch.

_Congrats on your first machine “learning” experiment! See you in Part 2.5!_
