+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.1: The Perceptron—Oldest Neural Network"
author = "Artintellica"
date = "2024-06-09"
+++

## Introduction

Welcome to **Module 3: Neural Networks—Building Blocks and Training**! Today
you’ll go back to the very roots of deep learning: the **perceptron**. Invented
in the 1950s, the perceptron is the simplest neural network—an algorithm for
classifying points by a weighted sum and a threshold. Every modern neural net
layer (including all of deep RL) is a direct descendant of this little
algorithm!

You will:

- Learn the perceptron's math and why it works.
- Implement the update rule from scratch in PyTorch.
- Train the perceptron on linearly separable and non-separable data.
- Visualize decision boundaries and analyze successes and failures.

Let’s awaken the ancestor of all modern AI.

---

## Mathematics: The Classic Perceptron

The perceptron is a binary classifier for input $\mathbf{x} \in \mathbb{R}^d$.
Given weights $\mathbf{w} \in \mathbb{R}^d$ and bias $b$:

$$
\text{output } \hat{y} = \begin{cases}
1 & \text{if } \mathbf{w}^\top\mathbf{x} + b > 0 \\
0 & \text{otherwise}
\end{cases}
$$

**Learning rule:**  
For each data point $(\mathbf{x}_i, y_i)$, if the prediction is incorrect:

- For $y_i = 1$: increase weights toward $\mathbf{x}_i$ (make output more likely
  $>0$)
- For $y_i = 0$: decrease weights toward $\mathbf{x}_i$ (make output more likely
  $<0$)

The parameter update after each sample:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \, (y_i - \hat{y}_i) \, \mathbf{x}_i \\
b \leftarrow b + \eta \, (y_i - \hat{y}_i)
$$

Where $\eta$ is the learning rate.

**Key property:** If the data is linearly separable, this algorithm finds a
separating hyperplane in finite steps.

---

## Explanation: How the Math Connects to Code

The perceptron models the **simplest neural network**—a single weighted sum
followed by a step function. At each step through the data, the perceptron
checks if its prediction matches the true label:

- If **correct**, do nothing.
- If **wrong**, it nudges the weights toward or away from the input, shifting
  the decision boundary accordingly. This process is repeated over several
  "epochs".

**Training loop:**

- Loop over the data; after each data point, update parameters only if
  misclassified.
- Repeat for several epochs, updating the boundary after each.

**Visualizing the boundary**:  
Since the perceptron produces a linear separator (a line in 2D), you can plot
this line after each epoch. On non-separable data, the perceptron will not
converge, highlighting an important limitation compared to modern neural nets.

---

## Python Demonstrations

### Demo 1: Implement the Perceptron Update Rule for a Binary Task

```python
import torch

def perceptron_step(x, y, w, b, lr=1.0):
    """
    x: input vector (d,)
    y: true label (0 or 1)
    w: weight vector (d,)
    b: bias (scalar)
    Returns updated w, b
    """
    # Prediction: if w^T x + b > 0: 1 else 0
    z = torch.dot(w, x) + b
    y_pred = 1.0 if z > 0 else 0.0
    if y != y_pred:
        # Update rule
        w = w + lr * (y - y_pred) * x
        b = b + lr * (y - y_pred)
    return w, b

# Simple test
w = torch.zeros(2)
b = 0.
x = torch.tensor([1.0, -1.0])
y = 1.0
w, b = perceptron_step(x, y, w, b)
print("Updated w/b:", w, b)
```

---

### Demo 2: Train the Perceptron on Linearly Separable Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate simple separable data
np.random.seed(0)
N = 40
X0 = np.random.randn(N, 2) + np.array([2,2])
X1 = np.random.randn(N, 2) + np.array([-2,-2])
X = np.concatenate([X0, X1])
y = np.concatenate([np.ones(N), np.zeros(N)])

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
w = torch.zeros(2)
b = 0.

epochs = 12
boundary_history = []

for epoch in range(epochs):
    for i in range(len(X)):
        w, b = perceptron_step(X_t[i], y_t[i], w, b, lr=0.7)
    boundary_history.append((w.clone(), b))

print("Final weights:", w, "Final bias:", b)
```

---

### Demo 3: Visualize Decision Boundary After Each Epoch

```python
def plot_perceptron_decision(X, y, boundary_history):
    plt.figure(figsize=(8,5))
    plt.scatter(X[y==0,0], X[y==0,1], color="orange", label="Class 0")
    plt.scatter(X[y==1,0], X[y==1,1], color="blue", label="Class 1")
    x_vals = np.array(plt.gca().get_xlim())
    for i, (w, b) in enumerate(boundary_history):
        # Line: w1*x + w2*y + b = 0 => y = (-w1*x - b)/w2
        if w[1].abs() > 1e-6:
            y_vals = (-w[0].item() * x_vals - b.item()) / w[1].item()
            plt.plot(x_vals, y_vals, alpha=0.3 + 0.7*i/len(boundary_history), label=f"Epoch {i+1}" if i==len(boundary_history)-1 else None)
    plt.legend(); plt.title("Perceptron Decision Boundary Evolution")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.show()

plot_perceptron_decision(X, y, boundary_history)
```

---

### Demo 4: Test on Non-Separable Data and Analyze Behavior

```python
# Make non-separable data (add noise and overlap)
np.random.seed(3)
N = 40
X0 = np.random.randn(N, 2) + np.array([1,1])
X1 = np.random.randn(N, 2) + np.array([2,2])
X_ns = np.concatenate([X0, X1])
y_ns = np.concatenate([np.zeros(N), np.ones(N)])

X_t_ns = torch.tensor(X_ns, dtype=torch.float32)
y_t_ns = torch.tensor(y_ns, dtype=torch.float32)
w_ns = torch.zeros(2)
b_ns = 0.
boundary_history_ns = []

for epoch in range(12):
    for i in range(len(X_ns)):
        w_ns, b_ns = perceptron_step(X_t_ns[i], y_t_ns[i], w_ns, b_ns, lr=0.7)
    boundary_history_ns.append((w_ns.clone(), b_ns))

plot_perceptron_decision(X_ns, y_ns, boundary_history_ns)
print("Final weights (non-separable):", w_ns, "Final bias:", b_ns)
```

---

## Exercises

### **Exercise 1:** Implement the Perceptron Update Rule for a Binary Task

- Write a function or code block that updates weights $w$ and bias $b$ given one
  sample $(x, y)$, as above.

---

### **Exercise 2:** Train the Perceptron on Linearly Separable Data

- Generate two clearly separate clusters in 2D.
- Train the perceptron for 10+ epochs using your update rule.
- Plot how decision boundary evolves over epochs.

---

### **Exercise 3:** Visualize Decision Boundary After Each Epoch

- After every epoch, store current $w$ and $b$.
- Plot all boundaries together (use transparency so they overlay).
- See how the classifier converges to a perfect separator.

---

### **Exercise 4:** Test on Non-Separable Data and Analyze Behavior

- Generate two noisy overlapping clusters.
- Train and plot as above.
- Does the perceptron converge? What does the final boundary look like?

---

### **Sample Starter Code for Exercises**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def perceptron_step(x, y, w, b, lr=1.0):
    z = torch.dot(w, x) + b
    y_pred = 1.0 if z > 0 else 0.0
    if y != y_pred:
        w = w + lr * (y - y_pred) * x
        b = b + lr * (y - y_pred)
    return w, b

# EXERCISE 1 already shown in demo

# EXERCISE 2: Linearly separable data
np.random.seed(1)
N = 50
X0 = np.random.randn(N, 2) + np.array([3,3])
X1 = np.random.randn(N, 2) + np.array([-3,-3])
X = np.vstack([X0, X1])
y = np.hstack([np.ones(N), np.zeros(N)])
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
w = torch.zeros(2)
b = 0.
history = []
for epoch in range(15):
    for i in range(len(X)):
        w, b = perceptron_step(X_t[i], y_t[i], w, b, lr=0.9)
    history.append((w.clone(), b))
# EXERCISE 3
def plot_perceptron(X, y, history):
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1')
    x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    for i, (w_, b_) in enumerate(history):
        if abs(w_[1]) > 1e-4:
            y_vals = (-w_[0].item() * x_vals - b_) / w_[1].item()
            plt.plot(x_vals, y_vals, alpha=(i+1)/len(history), label=f'Epoch {i+1}' if i == len(history)-1 else None)
    plt.xlabel('x1'); plt.ylabel('x2'); plt.legend(); plt.show()
plot_perceptron(X, y, history)

# EXERCISE 4: Non-separable case
np.random.seed(11)
X0_ns = np.random.randn(N, 2) + np.array([1,1])
X1_ns = np.random.randn(N, 2) + np.array([2,2])
X_ns = np.vstack([X0_ns, X1_ns])
y_ns = np.hstack([np.zeros(N), np.ones(N)])
X_t_ns = torch.tensor(X_ns, dtype=torch.float32)
y_t_ns = torch.tensor(y_ns, dtype=torch.float32)
w = torch.zeros(2)
b = 0.
history_ns = []
for epoch in range(15):
    for i in range(len(X_ns)):
        w, b = perceptron_step(X_t_ns[i], y_t_ns[i], w, b, lr=0.9)
    history_ns.append((w.clone(), b))
plot_perceptron(X_ns, y_ns, history_ns)
```

---

## Conclusion

You’ve brought the world’s first artificial neuron to life! Today, you:

- Implemented the classic perceptron and its update rule.
- Saw how learning linear boundaries works—and what happens when a simple model
  hits the limits of what it can separate.
- Gained skills in visualizing how decision boundaries evolve through learning.

**Up next:** We’ll move beyond perceptrons to networks of neurons—deep
learning’s true power—and you’ll implement neural nets by hand and with
PyTorch’s tools.

_The foundation is set—now let’s go deep! See you in Part 3.2._
