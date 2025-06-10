+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.3: Loss Functions and Cost Surfaces—Visual and Practical Intuition"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0070-rl-torch-23-loss"
+++

## Introduction

Welcome! In this installment of our RL-with-PyTorch series, we dive into the
**heart of learning**: **loss functions**. A model “learns” by minimizing a
loss—a quantitative measure of how wrong its predictions are. Understanding loss
functions and their _surfaces_ is your first step toward training not just
neural networks, but also policies and value functions in RL.

This post will help you:

- Grasp what loss functions are and _why_ they matter.
- Code and analyze the widely-used MSE (Mean Squared Error) and binary
  cross-entropy losses.
- Visualize loss surfaces to develop key intuition.
- See how losses handle outliers and shape model updates.

Let’s turn error into learning!

---

## Mathematics: Loss Functions and Surfaces

### What is a Loss Function?

At its core, a **loss function** $L(\hat{y}, y)$ quantifies the difference
between a prediction $\hat{y}$ and the true value $y$. Its minimization guides
training/optimization.

#### Mean Squared Error (MSE)

For regression or continuous outputs:

$$
L_\text{MSE}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2
$$

#### Binary Cross Entropy (BCE)

For binary classification ($y \in \{0, 1\}$, $\hat{y} \in (0, 1)$):

$$
L_\text{BCE}(\hat{y}, y) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$

The _shape_ of the loss surface determines how easily a model can be optimized!

---

## Python Demonstrations

### Demo 1: Implement MSE Loss Manually and with PyTorch

```python
import torch
import torch.nn.functional as F

y_true = torch.tensor([2.0, 3.5, 5.0])
y_pred = torch.tensor([2.5, 2.8, 4.6])

# Manual MSE
mse_manual = ((y_true - y_pred) ** 2).mean()
print("Manual MSE:", mse_manual.item())

# PyTorch MSE
mse_builtin = F.mse_loss(y_pred, y_true)
print("PyTorch MSE:", mse_builtin.item())
```

---

### Demo 2: Plot the Loss Surface for a Simple Linear Model

Let’s visualize $L(w) = \frac{1}{n}\sum (wx_i - y_i)^2$ for a fixed dataset and
see how loss changes as weight $w$ varies.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.1])  # Linear with small noise

w_vals = np.linspace(0, 3, 100)
loss_vals = [np.mean((w * x - y)**2) for w in w_vals]

plt.plot(w_vals, loss_vals)
plt.xlabel("w")
plt.ylabel("MSE Loss")
plt.title("Loss Surface for Linear Model y = w*x")
plt.grid(True)
plt.show()
```

Notice the bowl shape—this is characteristic of quadratic losses.

---

### Demo 3: Visualize Binary Cross-Entropy Loss as Function of Input

Let’s plot the BCE loss as a function of predicted probability $\hat{y}$ for
both possible labels $y \in \{0, 1\}$.

```python
p = np.linspace(1e-6, 1 - 1e-6, 200)
bce_y1 = -np.log(p)         # when y = 1
bce_y0 = -np.log(1 - p)     # when y = 0

plt.plot(p, bce_y1, label='y=1')
plt.plot(p, bce_y0, label='y=0')
plt.xlabel('Predicted probability ($\hat{y}$)')
plt.ylabel('BCE Loss')
plt.title('Binary Cross Entropy as a Function of $\hat{y}$')
plt.ylim(0, 6)
plt.legend()
plt.grid(True)
plt.show()
```

- Predicting $0.01$ when $y=1$ incurs a huge loss.
- BCE strongly punishes confident, wrong predictions.

---

### Demo 4: Compare How Different Losses Penalize Outliers

Let’s compare the effect of MSE and MAE (mean absolute error) on outliers.

```python
from matplotlib.ticker import MaxNLocator

y_true = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0])
errs   = np.linspace(-5, 10, 200)  # Error for the last (potential outlier) element

mse_vals = [F.mse_loss(torch.tensor([*y_true[:-1], y_true[-1] + e]), y_true).item() for e in errs]
mae_vals = [F.l1_loss (torch.tensor([*y_true[:-1], y_true[-1] + e]), y_true).item() for e in errs]

plt.plot(errs, mse_vals, label="MSE")
plt.plot(errs, mae_vals, label="MAE")
plt.xlabel("Outlier error")
plt.ylabel("Loss")
plt.title("Losses vs Outlier Error")
plt.legend()
plt.grid(True)
plt.show()
```

- **MSE** grows fast (quadratic)—outliers dominate the loss.
- **MAE** ($L_1$) is more forgiving—linear growth.

---

## Exercises

### **Exercise 1:** Implement Mean Squared Error (MSE) Loss Manually and with PyTorch

- Given $y_\text{true} = [1.5, 3.0, 4.0]$, $y_\text{pred} = [2.0, 2.5, 3.5]$,
  compute MSE manually.
- Verify it matches `torch.nn.functional.mse_loss`.

---

### **Exercise 2:** Plot the Loss Surface for a Simple Linear Model

- Let $x = [0, 1, 2, 3]$ and $y = [1, 2, 2, 4]$.
- For $w$ in $[-1, 3]$, compute loss $L(w) = \frac{1}{n}\sum (w x_i - y_i)^2$.
- Plot the loss curve as a function of $w$.

---

### **Exercise 3:** Visualize Binary Cross-Entropy Loss as a Function of Input

- Plot BCE loss vs. $\hat{y} \in [0.01, 0.99]$ for both $y=0$ and $y=1$.
- Try plotting $L_\text{BCE}$ for $y = 0.5$ (optional: when using soft labels).

---

### **Exercise 4:** Compare How Different Loss Functions Penalize Outliers

- For $y_\text{true} = [1, 1, 1, 10]$ and predicted
  $y_\text{pred} = [1, 1, 1, x]$,
- Vary $x$ from $0$ to $20$.
- Plot MSE and MAE as function of $x$.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# EXERCISE 1
y_true = torch.tensor([1.5, 3.0, 4.0])
y_pred = torch.tensor([2.0, 2.5, 3.5])
mse_manual = ((y_true - y_pred) ** 2).mean()
mse_torch = F.mse_loss(y_pred, y_true)
print("Manual MSE:", mse_manual.item())
print("PyTorch MSE:", mse_torch.item())

# EXERCISE 2
x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 2, 4])
ws = np.linspace(-1, 3, 100)
loss_curve = [np.mean((w*x - y)**2) for w in ws]
plt.plot(ws, loss_curve)
plt.xlabel("w"); plt.ylabel("MSE Loss")
plt.title("Loss Surface for Linear Model"); plt.grid(True); plt.show()

# EXERCISE 3
p = np.linspace(0.01, 0.99, 100)
bce_0 = -np.log(1 - p)
bce_1 = -np.log(p)
plt.plot(p, bce_0, label="y=0")
plt.plot(p, bce_1, label="y=1")
plt.xlabel("Predicted probability ($\hat{y}$)")
plt.ylabel("BCE Loss")
plt.legend(); plt.grid(True); plt.title("BCE Loss as Function of Prediction"); plt.show()

# EXERCISE 4
y_true = torch.tensor([1, 1, 1, 10], dtype=torch.float32)
x_pred = np.linspace(0, 20, 120)
mse_vals = [F.mse_loss(torch.tensor([1, 1, 1, val]), y_true).item() for val in x_pred]
mae_vals = [F.l1_loss(torch.tensor([1, 1, 1, val]), y_true).item() for val in x_pred]
plt.plot(x_pred, mse_vals, label="MSE")
plt.plot(x_pred, mae_vals, label="MAE")
plt.xlabel("Predicted outlier value")
plt.ylabel("Loss")
plt.legend(); plt.grid(True)
plt.title("Effect of Outlier on MSE vs MAE"); plt.show()
```

---

## Conclusion

Loss functions shape how models learn and what they prioritize. You’ve now:

- Coded MSE and BCE losses by hand and with PyTorch.
- Visualized loss surfaces and learned how their shapes impact optimization.
- Compared the robustness of different losses against outliers.

**Next:** You’ll fit your first **linear regression** model and visualize how
loss minimization leads to learning from data. Get ready to move from error
measurement to model training!

_See you in Part 2.4!_
