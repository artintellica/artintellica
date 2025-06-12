+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.6: Overfitting, Underfitting, and Regularization"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Now that you can build and train neural networks, you need to learn the art of **generalization**. A powerful model is not useful unless it does well on *new data*! In machine learning and RL, two classic “fail states” appear: **overfitting** and **underfitting**. Mastery of these—and their fixes—makes your models reliable in the wild.

In this post, you'll:

- Fit and compare models on small (overfit-prone) vs. large (underfit-prone) datasets.
- Add noise and see how it tempts your model to memorize rather than generalize.
- Apply **L2 regularization** ("weight decay") to reduce overfitting.
- Experimentally vary model complexity and measure the effect on accuracy.

---

## Mathematics: Overfitting, Underfitting, and L2 Regularization

- **Overfitting:** Model fits training set *too* closely, including noise. Train loss $\to 0$ but test loss $\gg 0$.
- **Underfitting:** Model is too simple to capture the true pattern. Both losses are high.
- **The solution:** Find a model that minimizes *test loss*, not just train loss.

### Losses (Train/Test)

Given:
- **Train loss:** $L_{\text{train}} = \frac{1}{N_\text{train}} \sum_{i=1}^{N_\text{train}} L(y_i, \hat{y}_i)$
- **Test loss:** $L_{\text{test}} = \frac{1}{N_\text{test}} \sum_{i=1}^{N_\text{test}} L(y_j, \hat{y}_j)$

### L2 Regularization (Weight Decay)

Add a penalty to loss for large weights:

$$
L_{\text{total}} = L_{\text{original}} + \lambda \sum_{k} w_k^2
$$

- $\lambda > 0$ is the regularization strength.
- Penalizes “complex” (high-magnitude) weights, encouraging simplicity.

---

## Explanation: How the Math Connects to Code

In code, you handle generalization by:

- **Splitting data:** Train your model on a subset (train set), but judge performance on *unseen* data (test set).
- **Overfitting/underfitting detection:** Plot *both* train and test loss as training progresses.
    - Overfitting: training loss plunges, test loss climbs.
    - Underfitting: both losses high or stagnant.
- **L2 regularization:** In PyTorch, set `weight_decay` in your optimizer, or manually add to the loss during training.
- **Model complexity:** Experiment with small/large nets or high/low degree polynomials, and see how flexibility relates to overfitting/underfitting.

You’ll see these phenomena visually—crucial for practical RL and ML development.

---

## Python Demonstrations

### Demo 1: Fit on Small vs. Large Dataset and Plot Train/Test Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate data: true function is nontrivial
def gen_data(N: int, noise: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    x: torch.Tensor = torch.linspace(-3, 3, N)
    y_true: torch.Tensor = torch.sin(x) + 0.5 * x
    y: torch.Tensor = y_true + noise * torch.randn(N)
    return x.unsqueeze(1), y

# Small and large splits
x_train, y_train = gen_data(20)
x_test, y_test = gen_data(100)

class TinyNet(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(1, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(1)

def train_and_eval(
    model: nn.Module,
    xtr: torch.Tensor, ytr: torch.Tensor,
    xte: torch.Tensor, yte: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    l2: float = 0.0
) -> tuple[list[float], list[float]]:
    opt: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    train_losses, test_losses = [], []
    for _ in range(epochs):
        model.train()
        pred: torch.Tensor = model(xtr)
        loss: torch.Tensor = F.mse_loss(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            test_pred = model(xte)
            test_loss = F.mse_loss(test_pred, yte)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
    return train_losses, test_losses

# Train on small set, test on large
net: TinyNet = TinyNet()
train_loss, test_loss = train_and_eval(net, x_train, y_train, x_test, y_test)
plt.plot(train_loss, label="Train loss (small set)")
plt.plot(test_loss, label="Test loss (large set)")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Train vs Test: Overfitting with Small Data")
plt.legend(); plt.show()
```

---

### Demo 2: Add Noise and Visualize Overfitting

```python
# High-noise small data
x_train2, y_train2 = gen_data(20, noise=1.0)
net2 = TinyNet()
train2, test2 = train_and_eval(net2, x_train2, y_train2, x_test, y_test)
plt.plot(train2, label="Train loss")
plt.plot(test2, label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Overfitting Worsens With Noise")
plt.legend(); plt.show()

# Visualize model fit
with torch.no_grad():
    plt.scatter(x_train2, y_train2, c='b', label='Train (noisy)')
    plt.plot(x_test, y_test, 'k:', label='True function')
    plt.plot(x_test, net2(x_test), 'r-', label='Model')
    plt.legend(); plt.title("Model Fit on Noisy Data"); plt.show()
```

---

### Demo 3: Apply L2 Regularization and Observe Effect

```python
# Try different L2 settings
losses_l2 = {}
for l2 in [0.0, 0.01, 0.1]:
    net_l2 = TinyNet()
    t_l, te_l = train_and_eval(net_l2, x_train2, y_train2, x_test, y_test, l2=l2)
    losses_l2[l2] = (t_l, te_l)

for l2, (t_l, te_l) in losses_l2.items():
    plt.plot(te_l, label=f"L2={l2}")
plt.xlabel("Epoch"); plt.ylabel("Test Loss")
plt.title("Test Loss by L2 Regularization")
plt.legend(); plt.show()
```

---

### Demo 4: Vary Model Complexity and Record Accuracy

```python
# Use model with varying hidden size
accs = []
sizes = [2, 8, 24, 64]
for h in sizes:
    net_c = TinyNet(hidden=h)
    t_l, te_l = train_and_eval(net_c, x_train, y_train, x_test, y_test, epochs=100)
    # R^2 as a measure of accuracy
    with torch.no_grad():
        preds = net_c(x_test)
        yte_mean = y_test.mean()
        ss_res = torch.sum((y_test - preds) ** 2)
        ss_tot = torch.sum((y_test - yte_mean) ** 2)
        r2 = 1 - ss_res/ss_tot
    accs.append(r2.item())
plt.plot(sizes, accs, marker='o')
plt.xlabel("Hidden Layer Size (Model Complexity)")
plt.ylabel("Test R^2 Score (Accuracy)")
plt.title("Effect of Model Complexity")
plt.grid(True); plt.show()
```

---

## Exercises

### **Exercise 1:** Fit a Model to Small and Large Datasets, Plot Train/Test Loss

- Generate two datasets: one small (e.g., $N=20$) and one large (e.g., $N=100$), from a nonlinear true function (e.g., $\sin(x) + 0.5x$).
- Train a small neural net on the small set, and evaluate loss on the large test set each epoch.
- Plot and compare train vs. test loss.

---

### **Exercise 2:** Add Noise to Data and Visualize Overfitting

- Add more noise to your training set.
- Retrain your model and plot losses.
- Visualize predictions on the test set and training data.

---

### **Exercise 3:** Apply L2 Regularization and Observe the Effect

- Retrain your model on noisy data, varying `weight_decay` in the optimizer (e.g., 0, 0.01, 0.1).
- Compare test set loss over epochs.

---

### **Exercise 4:** Vary Model Complexity and Record Accuracy

- For several hidden layer widths (e.g., 2, 8, 16, 32, 64), train and evaluate models.
- For each, compute test $R^2$ or accuracy.
- Plot test accuracy vs. model size.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def gen_data(N: int, noise: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    x: torch.Tensor = torch.linspace(-3, 3, N)
    y_true: torch.Tensor = torch.sin(x) + 0.5 * x
    y: torch.Tensor = y_true + noise * torch.randn(N)
    return x.unsqueeze(1), y

class TinyNet(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(1, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(1)

def train_and_eval(
    model: nn.Module,
    xtr: torch.Tensor, ytr: torch.Tensor,
    xte: torch.Tensor, yte: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    l2: float = 0.0
) -> tuple[list[float], list[float]]:
    opt: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    train_losses, test_losses = [], []
    for _ in range(epochs):
        model.train()
        pred: torch.Tensor = model(xtr)
        loss: torch.Tensor = F.mse_loss(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            test_pred = model(xte)
            test_loss = F.mse_loss(test_pred, yte)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
    return train_losses, test_losses

# EXERCISE 1
x_train, y_train = gen_data(20)
x_test, y_test = gen_data(100)
net = TinyNet()
train_loss, test_loss = train_and_eval(net, x_train, y_train, x_test, y_test)
plt.plot(train_loss, label="Train")
plt.plot(test_loss, label="Test")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Overfitting on Small Data"); plt.show()

# EXERCISE 2
x_train2, y_train2 = gen_data(20, noise=1.0)
net2 = TinyNet()
train2, test2 = train_and_eval(net2, x_train2, y_train2, x_test, y_test)
plt.plot(train2, label="Train"); plt.plot(test2, label="Test"); plt.legend()
plt.title("Overfitting Worse With Noise"); plt.show()
with torch.no_grad():
    plt.scatter(x_train2, y_train2, c='b', label='Train (noisy)')
    plt.plot(x_test, y_test, 'k:', label='True')
    plt.plot(x_test, net2(x_test), 'r-', label='Model')
    plt.legend(); plt.title("Prediction on Noisy Data"); plt.show()

# EXERCISE 3
losses_l2 = {}
for l2 in [0.0, 0.01, 0.1]:
    net_l2 = TinyNet()
    t_l, te_l = train_and_eval(net_l2, x_train2, y_train2, x_test, y_test, l2=l2)
    losses_l2[l2] = (t_l, te_l)
for l2, (t_l, te_l) in losses_l2.items():
    plt.plot(te_l, label=f"L2={l2}")
plt.xlabel("Epoch"); plt.ylabel("Test Loss"); plt.legend(); plt.title("L2 Regularization Effect"); plt.show()

# EXERCISE 4
sizes = [2, 8, 32, 64]
accs = []
for h in sizes:
    net_c = TinyNet(hidden=h)
    t_l, te_l = train_and_eval(net_c, x_train, y_train, x_test, y_test, epochs=90)
    with torch.no_grad():
        preds = net_c(x_test)
        yte_mean = y_test.mean()
        ss_res = torch.sum((y_test - preds) ** 2)
        ss_tot = torch.sum((y_test - yte_mean) ** 2)
        r2 = 1 - ss_res/ss_tot
    accs.append(r2.item())
plt.plot(sizes, accs, marker='o')
plt.xlabel("Hidden Layer Size (Model Complexity)")
plt.ylabel("Test R^2 Score")
plt.title("Accuracy vs. Model Complexity")
plt.grid(True); plt.show()
```

---

## Conclusion

You have now:

- Diagnosed overfitting and underfitting by measuring and plotting train/test loss
- Seen how noise and small datasets make models memorize instead of generalize
- Applied L2 regularization to reduce model complexity and overfitting  
- Explored how model size affects learning and accuracy

**Next:** You’ll learn more regularization tricks—Dropout, L2, and how PyTorch makes regularized training fast and easy.

*Keep experimenting: these skills give your models staying power in the real world! See you in Part 3.7!*
