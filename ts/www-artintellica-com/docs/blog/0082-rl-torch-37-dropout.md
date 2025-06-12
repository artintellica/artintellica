+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.7: Dropout, L2, and Other Regularization in PyTorch"
author = "Artintellica"
date = "2024-06-12"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0082-rl-torch-37-dropout"
+++

## Introduction

By now, you’ve learned why **regularization** is essential in machine learning:
without it, neural networks can memorize their training data and fail to
generalize. PyTorch offers built-in tools for regularization, from **dropout
layers** to **L2 (weight decay)** and more. In this post you’ll:

- Implement dropout and weight decay in PyTorch models.
- Visualize how regularization improves model generalization.
- Compare models with and without regularization.
- Experiment with dropout rates and interpret their effects.

---

## Mathematics: Dropout, L2, and Regularization Concepts

**L2 Regularization (Weight Decay):**

- Adds a penalty term to the loss that suppresses large weights:
  $$
  L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i=1}^{N} w_i^2
  $$
- PyTorch implements this via the `weight_decay` parameter in optimizers.

**Dropout:**

- In training, randomly sets a fraction $p$ of activations to zero at each layer
  per batch:
  $$
  h_{\text{drop}} = \mathbf{d} \odot h, \quad \mathbf{d} \sim \text{Bernoulli}(1-p)
  $$
- At test time, all neurons are used (scaling is applied automatically).
- Dropout acts as model bagging—makes the network robust to missing
  features/nodes.

**The effect:** Both L2 and dropout combat overfitting by “simplifying” the
model and preventing reliance on any one parameter or path.

---

## Explanation: How the Math Connects to Code

In practice with PyTorch:

- **L2 Regularization:** Set `weight_decay=λ` in your optimizer. PyTorch
  automatically adds $λ \sum w_i^2$ to the loss.
- **Dropout:** Insert `nn.Dropout(p)` layers after any linear or activation
  layer. During training, each forward pass zeroes out different random units—at
  inference, dropout is disabled.
- **Validation splitting:** Hold out a fraction of your data as the "validation
  set" (unseen during training) so you can meaningfully evaluate generalization
  under various regularization schemes.
- PyTorch handles all state switching for Dropout between `.train()` and
  `.eval()`, so you get correct behavior during training and test.

Using these tools, you’ll see how accuracy and generalization improve or degrade
as you tweak regularization strength or dropout rate.

---

## Python Demonstrations

### Demo 1: Add Dropout Layers To a Network and Plot Training/Validation Accuracy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Synthetic dataset (nonlinear boundary)
torch.manual_seed(7)
N: int = 200
X: torch.Tensor = torch.randn(N, 2)
y: torch.Tensor = ((X[:,0]**2 + 0.8*X[:,1]) > 0.5).long()

class DropNet(nn.Module):
    def __init__(self, hidden: int = 32, pdrop: float = 0.5) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.drop1: nn.Dropout = nn.Dropout(pdrop)
        self.fc2: nn.Linear = nn.Linear(hidden, hidden)
        self.drop2: nn.Dropout = nn.Dropout(pdrop)
        self.fc3: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

# Split into train/val
train_idx = torch.arange(0, int(0.7*N))
val_idx = torch.arange(int(0.7*N), N)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

net: DropNet = DropNet(hidden=32, pdrop=0.5)
optimizer = torch.optim.Adam(net.parameters(), lr=0.08)
loss_fn = nn.CrossEntropyLoss()
train_acc: list[float] = []
val_acc: list[float] = []

for epoch in range(80):
    net.train()
    logits = net(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred_train = logits.argmax(dim=1)
    acc_t = (pred_train == y_train).float().mean().item()
    train_acc.append(acc_t)

    net.eval()
    with torch.no_grad():
        val_logits = net(X_val)
        pred_val = val_logits.argmax(dim=1)
        acc_v = (pred_val == y_val).float().mean().item()
        val_acc.append(acc_v)

plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Dropout: Train vs. Val Accuracy")
plt.legend(); plt.grid(True); plt.show()
```

---

### Demo 2: Use Weight Decay (L2) with an Optimizer

```python
net_l2: DropNet = DropNet(hidden=32, pdrop=0.5)
optimizer_l2 = torch.optim.Adam(net_l2.parameters(), lr=0.08, weight_decay=0.02)
train_acc_l2: list[float] = []
val_acc_l2: list[float] = []

for epoch in range(80):
    net_l2.train()
    logits = net_l2(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

    pred_train = logits.argmax(dim=1)
    acc_t = (pred_train == y_train).float().mean().item()
    train_acc_l2.append(acc_t)

    net_l2.eval()
    with torch.no_grad():
        val_logits = net_l2(X_val)
        pred_val = val_logits.argmax(dim=1)
        acc_v = (pred_val == y_val).float().mean().item()
        val_acc_l2.append(acc_v)

plt.plot(train_acc, label="Train Acc (Dropout)")
plt.plot(val_acc, label="Val Acc (Dropout)")
plt.plot(train_acc_l2, '--', label="Train Acc (Dropout+L2)")
plt.plot(val_acc_l2, '--', label="Val Acc (Dropout+L2)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Effect of Dropout and L2 (weight decay)")
plt.legend(); plt.grid(True); plt.show()
```

---

### Demo 3: Compare Models With and Without Regularization

```python
# Model with no regularization
class PlainNet(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, hidden)
        self.fc3: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net_plain = PlainNet()
optimizer_plain = torch.optim.Adam(net_plain.parameters(), lr=0.08)
train_acc_plain: list[float] = []
val_acc_plain: list[float] = []

for epoch in range(80):
    net_plain.train()
    logits = net_plain(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_plain.zero_grad()
    loss.backward()
    optimizer_plain.step()

    pred_train = logits.argmax(dim=1)
    acc_t = (pred_train == y_train).float().mean().item()
    train_acc_plain.append(acc_t)

    net_plain.eval()
    with torch.no_grad():
        val_logits = net_plain(X_val)
        pred_val = val_logits.argmax(dim=1)
        acc_v = (pred_val == y_val).float().mean().item()
        val_acc_plain.append(acc_v)

plt.plot(train_acc_plain, label="Train Acc (No Reg)")
plt.plot(val_acc_plain, label="Val Acc (No Reg)")
plt.plot(train_acc, label="Train Acc (Dropout)")
plt.plot(val_acc, label="Val Acc (Dropout)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Regularization vs. No Regularization")
plt.legend(); plt.grid(True); plt.show()
```

---

### Demo 4: Experiment With Different Dropout Rates and Interpret Results

```python
dropout_rates = [0.0, 0.3, 0.5, 0.7]
acc_by_rate = {}
for p in dropout_rates:
    net_p = DropNet(hidden=32, pdrop=p)
    optimizer_p = torch.optim.Adam(net_p.parameters(), lr=0.08)
    val_accs = []
    for epoch in range(60):
        net_p.train()
        logits = net_p(X_train)
        loss = loss_fn(logits, y_train)
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_p.step()
        net_p.eval()
        with torch.no_grad():
            val_logits = net_p(X_val)
            pred_val = val_logits.argmax(dim=1)
            acc_v = (pred_val == y_val).float().mean().item()
            val_accs.append(acc_v)
    acc_by_rate[p] = val_accs

for p, val_accs in acc_by_rate.items():
    plt.plot(val_accs, label=f"Dropout={p}")
plt.xlabel("Epoch"); plt.ylabel("Val Accuracy")
plt.title("Validation Accuracy vs. Dropout Rate")
plt.legend(); plt.grid(True); plt.show()
```

---

## Exercises

### **Exercise 1:** Add Dropout Layers To a Network and Plot Training/Validation Accuracy

- Build a neural network for 2D classification with at least one `nn.Dropout()`
  layer.
- Split your data into train and validation sets.
- Plot accuracies for both sets over training.

---

### **Exercise 2:** Use Weight Decay With an Optimizer

- Train the same network with and without `weight_decay` (e.g. $0.02$) in the
  optimizer.
- Plot and compare validation performance.

---

### **Exercise 3:** Compare Models With and Without Regularization

- Train and plot accuracy (or loss) for:
  1. No regularization,
  2. Dropout,
  3. Dropout + L2.
- Compare overfitting and generalization visually.

---

### **Exercise 4:** Experiment With Different Dropout Rates and Interpret Results

- Vary dropout (from 0 to 0.7+).
- Plot validation accuracy curves, and interpret how too little/too much dropout
  affects training.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Data (Nonlinear)
torch.manual_seed(11)
N = 180
X = torch.randn(N, 2)
y = ((X[:,0]**2 - 0.7*X[:,1]) > 0.3).long()
train_idx = torch.arange(0, int(0.6*N))
val_idx = torch.arange(int(0.6*N), N)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# EXERCISE 1
class RegNet(nn.Module):
    def __init__(self, hidden: int = 28, pdrop: float = 0.4) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(2, hidden)
        self.drop: nn.Dropout = nn.Dropout(pdrop)
        self.fc2: nn.Linear = nn.Linear(hidden, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
net = RegNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.07)
loss_fn = nn.CrossEntropyLoss()
train_accs, val_accs = [], []
for epoch in range(60):
    net.train()
    logits = net(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(dim=1) == y_train).float().mean().item()
    train_accs.append(acc)
    net.eval()
    with torch.no_grad():
        val_logits = net(X_val)
        acc_v = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        val_accs.append(acc_v)
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Dropout Regularization"); plt.legend(); plt.grid(); plt.show()

# EXERCISE 2
net_l2 = RegNet()
optimizer_l2 = torch.optim.Adam(net_l2.parameters(), lr=0.07, weight_decay=0.02)
val_accs_l2 = []
for epoch in range(60):
    net_l2.train()
    logits = net_l2(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()
    net_l2.eval()
    with torch.no_grad():
        val_logits = net_l2(X_val)
        acc_v = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        val_accs_l2.append(acc_v)
plt.plot(val_accs, label="Dropout Only")
plt.plot(val_accs_l2, label="Dropout + L2")
plt.title("Weight Decay in Optimizer"); plt.xlabel("Epoch"); plt.ylabel("Val Accuracy")
plt.legend(); plt.grid(); plt.show()

# EXERCISE 3
# See Demo 3 for full comparison. Try repeating with and without dropout/L2.

# EXERCISE 4
rates = [0.0, 0.2, 0.5, 0.7]
for p in rates:
    regnet = RegNet(pdrop=p)
    optimizer = torch.optim.Adam(regnet.parameters(), lr=0.07)
    val_accs_this = []
    for epoch in range(40):
        regnet.train()
        logits = regnet(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        regnet.eval()
        with torch.no_grad():
            val_logits = regnet(X_val)
            acc_val = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            val_accs_this.append(acc_val)
    plt.plot(val_accs_this, label=f"Dropout={p}")
plt.title("Dropout Rate Effect"); plt.xlabel("Epoch"); plt.ylabel("Val Acc")
plt.legend(); plt.grid(); plt.show()
```

---

## Conclusion

You have learned:

- How to implement and tune dropout, L2, and other regularization in PyTorch
- How regularization impacts overfitting and generalization in real networks
- That model robustness is often about what you _leave out,_ not just what you
  put in!

**Next:** You’ll move from datasets to real computer vision with a classic
benchmark—training a (shallow) neural net on MNIST.

_Keep experimenting: regularization is your best defense against overfitting in
RL and deep learning. See you in Part 3.8!_
