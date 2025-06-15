+++
title = "Learn the Training Loop with PyTorch, Part 1.3: Batch vs. Stochastic Gradient Descent"
author = "Artintellica"
date = "2025-06-15"
model = "grok-3"
userDelimiter = "**USER:**"
assistantDelimiter = "**ASSISTANT:**"
+++

**USER:**

i need your help to write the next blog post in a series i am producing called "Learn the Training Loop with PyTorch".

first, let me give you the outline of the series:

---

## **Module 1: The Elementary Training Loop**

**Goal:** Build foundational intuition by hand-coding a simple training loop from scratch.

**1. Introduction: What is a Training Loop?**
   - Definition and importance  
   - Loss, optimization, and learning in ML

**2. The Simplest Case: Linear Regression**
   - What is linear regression?
   - The mean squared error loss
   - Deriving the update rules manually

**3. Batch vs. Stochastic Gradient Descent**
   - Differences and practical implications  
   - Hands-on: See both in Python/numpy

**4. Visualizing the Loss Landscape**
   - Plotting loss as a function of parameters
   - Demo: Watching convergence by hand

**5. Numerical vs. Analytical Gradients**
   - How are gradients computed?  
   - Manual calculation and confirmation with numerical gradient

**6. Recap and Key Takeaways**

---

## **Module 2: The Training Loop for Neural Networks**

**Goal:** Extend the basic loop to more powerful models; introduce neural network concepts.

**1. From Linear to Nonlinear: Why Neural Networks?**
   - Multi-layer perceptrons (MLPs)  
   - Activation functions (ReLU, sigmoid, tanh)

**2. Forward and Backward Passes**
   - The chain rule of differentiation  
   - Backpropagation intuition (no need for full rigor yet)

**3. Implementing a Simple Neural Net from Scratch**
   - Single hidden layer network in numpy/Python
   - Coding forward, loss, backward, and parameter updates

**4. The Role of Activations**
   - Intuitive explanation and code examples
   - Effect on learning and expressiveness

**5. Mini-batching and Data Pipelines**
   - How batching speeds up and stabilizes learning

**6. Regularization and Overfitting**
   - Techniques: L2, dropout (conceptual intro)

**7. Recap: Comparing Our Simple Network with Linear Regression**

---

## **Module 3: Advanced Training Loops and Modern Tricks**

**Goal:** Connect foundational understanding to the large-scale models used in industry.

**1. Optimization Algorithms Beyond SGD**
   - Adam, RMSProp, etc.: How they work, why they're useful

**2. Learning Rate Scheduling**
   - How/why learning rates are changed during training  
   - Demo: Effect of learning rate on convergence and results

**3. Weight Initialization**
   - Why it matters, common strategies

**4. Deeper Networks and Backprop Challenges**
   - Vanishing/exploding gradients  
   - Solutions: normalization, skip connections

**5. Large-Scale Training: Data Parallelism and Hardware**
   - How the same training loop runs efficiently on GPUs and across many machines

**6. Monitoring and Debugging the Training Loop**
   - Visualizing loss/accuracy, diagnosing common problems

**7. Modern Regularization and Generalization Techniques**
   - Early stopping, batch/layer normalization, data augmentation (conceptual intro)

**8. The Training Loop in Practice: Case Studies**
   - High-level overview of how OpenAI/Google/Meta might train LLMs: what’s the same, what’s different.
   - What "scaling up" really means: dataset, parameters, hardware, tricks.

**9. Conclusion: What's Next After the Training Loop?**
   - Discussion: fine-tuning, transfer learning, and unsupervised/self-supervised methods.

---

### **General Tips for Each Module:**

- **Start simple, build incrementally:** Keep code simple at first, then show how to layer complexity.
- **Accompany every lesson with practical code examples (preferably in Jupyter notebooks).**
- **Focus on intuition:** Use diagrams, plots, and hands-on demo to make concepts tangible.
- **Summarize and review regularly:** Add checklists or quizzes at the end of each module.

---

great. now, for each blog post, i am hoping to follow the following structure:

- the course is called "Learn the Training Loop with PyTorch"
- always provide a brief introduction to the topic at hand
- then always provide a very simple "ELI5" explanation of the topic, that theortically anyone could understand
- then provide the mathematical foundations. for math, always use dollar delimited LaTeX, and always provide a brief explanation of the math. for instance, $\mathbf{y} = \mathbf{X}\mathbf{w}$ is the equation for linear regression, where $\mathbf{y}$ is the output vector, $\mathbf{X}$ is the input matrix, and $\mathbf{w}$ is the weight vector.
- for block-level math, always use two dollar signs, like this:
  $$
  \mathbf{y} = \mathbf{X}\mathbf{w}
  $$
- after the mathematical foundations, such as definitions, then always provide an in-depth explanation, covering the topic in detail, and expanding on the explanation of the math if necessary.
- for other formatting, use markdown like headings, bullet points, and code blocks to make the content readable. do not use HTML tags.
- after the math is complete, then provide one or more demonstrations of the topic using PyTorch. if possible, always use PyTorch, not numpy, just to keep it simple and consistnt. you can also use matplotlib to visualize the results.
- for all python code, always use type hints, and, if necessary, describe installation instructions using "uv" (not pip).
- then, create a series of exercises, somewhere from 1 - 4 total exercises that cover the material. please provide the soruce code for each exercise to make sure the student can easily compare their solution to the correct answer. do not hide the exercises and do not put HTML in the exercises.
- finally, provide a summary of the topic, and a list of key takeaways.

that's it, now can you please write blog post 1.3? the description is:

**3. Batch vs. Stochastic Gradient Descent**
   - Differences and practical implications  
   - Hands-on: See both in Python/numpy

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 1.3: Batch vs. Stochastic Gradient Descent

---

### Introduction

When training machine learning models, how you update your parameters can make a big difference in how fast and how well your model learns. Two of the most fundamental methods for updating parameters are **Batch Gradient Descent** and **Stochastic Gradient Descent (SGD)**.

In this lesson, you'll learn what these terms mean, why they matter, and how they shape the way we train models in practice. We'll break it down simply, build up some mathematical intuition, and then try them both out in PyTorch!

---

### ELI5: What Are Batch and Stochastic Gradient Descent?

Imagine you’re learning to shoot basketball free throws. Every evening, you shoot 100 balls. 

- **Batch Gradient Descent** is like watching a video of **all 100 shots**, then making one big adjustment to your shooting technique based on the average of all your mistakes.
- **Stochastic Gradient Descent** is like looking at **each shot in turn** and making a little adjustment after every single shot.

**Batches** make your learning stable, but slow to react. **Stochastic** updates are fast and noisy—they can wobble around a lot, but sometimes get you better results faster! There’s also a middle ground: **mini-batch**, which uses small groups instead of one shot or all shots.

---

### Mathematical Foundations

#### Core Update Rule

In machine learning, gradients tell us how to update our weights to reduce the error (loss). Given a loss function $L(\mathbf{w})$, the basic weight update rule is:

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \nabla_{\mathbf{w}} L
$$

where:
- $\mathbf{w}$ is the model's parameters (weights)
- $\eta$ is the learning rate (how big a step to take)
- $\nabla_{\mathbf{w}} L$ is the gradient of the loss with respect to the weights

#### What’s the Difference?

- **Batch Gradient Descent** uses **all data** to compute the gradient:
  $$
  \nabla_{\mathbf{w}} L = \frac{1}{N} \sum_{i=1}^N \nabla_{\mathbf{w}} \ell(\mathbf{x}_i, y_i; \mathbf{w})
  $$
  where $N$ is the total number of samples.

- **Stochastic Gradient Descent** uses **one sample** at a time:
  $$
  \nabla_{\mathbf{w}} L \approx \nabla_{\mathbf{w}} \ell(\mathbf{x}_j, y_j; \mathbf{w})
  $$
  for randomly chosen $j$ at each step.

- **Mini-batch Gradient Descent** uses a subset ($B$ samples):
  $$
  \nabla_{\mathbf{w}} L \approx \frac{1}{B} \sum_{i=1}^B \nabla_{\mathbf{w}} \ell(\mathbf{x}_i, y_i; \mathbf{w})
  $$

---

### In-Depth Explanation

#### Batch Gradient Descent (BGD)

BGD computes **the average gradient over the entire dataset** and performs one update per epoch (one sweep through data). This means each update is precise but requires you to see all the data before making any change. Useful when data fits in memory and you want smooth progress, but can be slow for large datasets.

- **Pros:** Stable updates, good convergence properties.
- **Cons:** Can be slow, not scalable for big datasets.

#### Stochastic Gradient Descent (SGD)

SGD computes the gradient and updates the weights **using only a single data point at a time**. This means you can make many small updates, which injects noise and randomness into your steps—which can help avoid bad local minima, but can make convergence bouncy.

- **Pros:** Fast updates, works well with large or streaming data, better for escaping shallow minima.
- **Cons:** High variance, sometimes less stable, might not converge smoothly.

#### Mini-batch Gradient Descent

A compromise: use small batches (like 16, 32, or 64 samples). This combines the stability of batch methods with the speed of stochastic methods. It’s the default in modern deep learning!

---

### PyTorch Demos

Let's see both methods in action with a simple **linear regression** problem.

#### Setup: Install Dependencies

First, to make sure you have PyTorch and matplotlib installed, run:

```bash
uv pip install torch matplotlib
```

---

#### 1. Batch Gradient Descent in PyTorch

```python
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List

# Generate dummy regression data
def generate_data(n: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    X = torch.linspace(0, 1, n).unsqueeze(1)
    y = 2 * X + 1 + 0.1 * torch.randn_like(X)  # y = 2x + 1 + noise
    return X, y

# Linear regression batch update
def train_batch(X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.1) -> List[float]:
    # Initialize weights and bias
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    losses = []

    for epoch in range(epochs):
        # Forward
        y_pred = X * w + b
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()

    return losses

X, y = generate_data()
losses_batch = train_batch(X, y, epochs=50, lr=0.5)

plt.plot(losses_batch, label="Batch Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Batch Gradient Descent Loss Curve")
plt.legend()
plt.show()
```

---

#### 2. Stochastic Gradient Descent in PyTorch

```python
def train_stochastic(X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.1) -> List[float]:
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    n = X.size(0)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in perm:
            xi = X[i]
            yi = y[i]
            # Forward
            y_pred = xi * w + b
            loss = (y_pred - yi) ** 2
            epoch_loss += loss.item()

            # Backward
            loss.backward()

            # Update
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad

            w.grad.zero_()
            b.grad.zero_()
        losses.append(epoch_loss / n)
    return losses

losses_sgd = train_stochastic(X, y, epochs=50, lr=0.05)

plt.plot(losses_sgd, label="Stochastic Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stochastic Gradient Descent Loss Curve")
plt.legend()
plt.show()
```

---

#### 3. Mini-batch Gradient Descent in PyTorch

```python
def train_minibatch(X: torch.Tensor, y: torch.Tensor, batch_size: int = 16, epochs: int = 100, lr: float = 0.1) -> List[float]:
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    n = X.size(0)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xi = X[idx]
            yi = y[idx]
            # Forward
            y_pred = xi * w + b
            loss = ((y_pred - yi) ** 2).mean()
            epoch_loss += loss.item() * xi.size(0)

            # Backward
            loss.backward()

            # Update
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad

            w.grad.zero_()
            b.grad.zero_()
        losses.append(epoch_loss / n)
    return losses

losses_minibatch = train_minibatch(X, y, batch_size=16, epochs=50, lr=0.1)

plt.plot(losses_batch, label="Batch")
plt.plot(losses_sgd, label="Stochastic")
plt.plot(losses_minibatch, label="Mini-batch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves: Batch vs. Stochastic vs. Mini-batch")
plt.legend()
plt.show()
```

#### Quick Comparison

You'll see that:

- **Batch**: Smoother curve, slower per update.
- **Stochastic**: More jagged curve, sometimes faster at reaching low loss.
- **Mini-batch**: Smooth *and* fast! Just right for modern deep learning.

---

### Exercises

#### 1. **Exercise:** Change the batch size in the mini-batch code and observe the effect.

```python
# Try batch_size = 1 (should behave like SGD), and batch_size = X.size(0) (should behave like Batch GD)
losses_bs1 = train_minibatch(X, y, batch_size=1, epochs=50, lr=0.1)
losses_bsfull = train_minibatch(X, y, batch_size=X.size(0), epochs=50, lr=0.1)

plt.plot(losses_bs1, label="Batch size = 1 (SGD)")
plt.plot(losses_bsfull, label="Full Batch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Effect of Batch Size")
plt.legend()
plt.show()
```

#### 2. **Exercise:** Try a much larger dataset (e.g., 10,000 samples) and compare training times and convergence with batch vs. mini-batch.

```python
X_large, y_large = generate_data(10_000)

import time

start = time.time()
losses_large_batch = train_batch(X_large, y_large, epochs=10, lr=0.5)
print("Batch GD Time:", time.time() - start)

start = time.time()
losses_large_mb = train_minibatch(X_large, y_large, batch_size=64, epochs=10, lr=0.5)
print("Mini-batch GD Time:", time.time() - start)
```

#### 3. **Exercise:** Add random noise to your data and see which method finds a 'flatter' loss curve.

```python
X_noisy, y_noisy = generate_data(100)
y_noisy += 0.5 * torch.randn_like(y_noisy)

losses_batch_noisy = train_batch(X_noisy, y_noisy, epochs=50, lr=0.5)
losses_sgd_noisy = train_stochastic(X_noisy, y_noisy, epochs=50, lr=0.05)
losses_minibatch_noisy = train_minibatch(X_noisy, y_noisy, batch_size=16, epochs=50, lr=0.1)

plt.plot(losses_batch_noisy, label="Batch")
plt.plot(losses_sgd_noisy, label="Stochastic")
plt.plot(losses_minibatch_noisy, label="Mini-batch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves with Noisy Data")
plt.legend()
plt.show()
```

---

### Summary

- **Batch Gradient Descent** uses all data for each update, giving accurate but slower and less flexible training.
- **Stochastic Gradient Descent (SGD)** makes fast, noisy updates based on single samples, great for large data and escaping local minima but potentially unstable.
- **Mini-batch Gradient Descent** strikes the best balance, combining the stability of batch with the speed of stochastic—it’s the standard in modern deep learning frameworks like PyTorch.

---

### Key Takeaways

- The difference between batch, stochastic, and mini-batch methods is **how much data you use at each weight update**.
- **Mini-batching** is practical and usually the best choice for training neural networks.
- PyTorch makes it simple to experiment with these approaches—**try changing batch sizes and see what happens!**
- Next up: You'll start building simple neural networks, where these batching concepts become even more powerful.

---

Experiment, learn, and see how the training loop comes alive in PyTorch!

**USER:**


