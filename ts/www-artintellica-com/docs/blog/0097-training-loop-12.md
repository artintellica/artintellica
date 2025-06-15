+++
title = "Learn the Training Loop with PyTorch, Part 1.2: The Simplest Case: Linear Regression"
author = "Artintellica"
date = "2025-06-15"
model = "o3"
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

that's it, now can you please write blog post 1.2? the description is:

**2. The Simplest Case: Linear Regression**
   - What is linear regression?
   - The mean squared error loss
   - Deriving the update rules manually

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 1.2: **The Simplest Case – Linear Regression**

---

### Introduction

Welcome to the second lesson in our “Learn the Training Loop with PyTorch” series! In this post, we’ll dive into the most classic example of machine learning: **linear regression**. By understanding this simple case, you’ll build a rock-solid foundation for everything to come—neural networks, deep learning, and beyond.

We’ll answer:
- What is linear regression?
- What is the mean squared error (MSE) loss?
- How do we calculate parameter updates **by hand**?

We'll also put all this into practice with clear PyTorch code and exercises so you can see theory in action.

---

## ELI5: What is Linear Regression? 

Imagine you want to predict someone’s height just by knowing their age. If you plot a bunch of people’s ages and heights on graph paper, but the dots look like they roughly follow a straight line, you can guess someone’s height by “drawing a line through the dots.” **Linear regression** is just a method for drawing the best straight line through the data, so you can predict *Y* (height) from *X* (age).

---

## Mathematical Foundations

Let’s formalize linear regression and see where loss and updates come from.

### The Model: Predicting with a Line

If you have $N$ data points, each with one feature $x$ and target $y$, the simplest linear regression model predicts:

$$
\hat{y}_i = w x_i + b
$$

Where:
- $w$ = the *weight* (slope of the line)
- $b$ = the *bias* (intercept; where the line crosses the y-axis)
- $x_i$ = input for the $i$th datapoint
- $\hat{y}_i$ = model's prediction

For many datapoints at once, with vectors:

$$
\hat{\mathbf{y}} = w \mathbf{x} + b
$$

or, with matrices (for more than 1 feature):

$$
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b
$$

Where:
- $\mathbf{X}$ is the $N \times D$ matrix of $N$ samples and $D$ features,
- $\mathbf{w}$ is the $D$-dimensional weight vector.

---

### The Loss: Mean Squared Error (MSE)

The most common way to measure how “off” our predictions are is **Mean Squared Error**:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

- $(\hat{y}_i - y_i)^2$ is the squared difference (error) between our prediction and the actual value.
- Averaging over all points gives us a single value: *the lower, the better!*

---

### Learning: Deriving the Update Rules

We want to find $w$ and $b$ that make the loss as small as possible.
To do this, we use **gradient descent**:  
1. Compute how the loss changes as we change $w$ and $b$ (the gradients)
2. Move (“step”) $w$ and $b$ in the direction that *decreases* the loss.

Let's compute the gradients for $w$ and $b$ (for 1 feature, 1 output):

- For $w$:

    $$
    \frac{\partial \text{MSE}}{\partial w} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)x_i
    $$

- For $b$:

    $$
    \frac{\partial \text{MSE}}{\partial b} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)
    $$

**Gradient descent update:**

At each iteration,

- $w \leftarrow w - \eta \frac{\partial \text{MSE}}{\partial w}$
- $b \leftarrow b - \eta \frac{\partial \text{MSE}}{\partial b}$

where $\eta$ is the *learning rate* – a small positive value controlling our step size.

---

## In-Depth Explanation

Let’s break it down:

- **Linear regression** finds the line (or hyperplane) that best fits your data by minimizing the average of squared prediction errors.
- **MSE** penalizes big mistakes much more than small ones (squaring makes outliers matter more).
- **Gradient descent** is an iterative process: we compute the direction in which the loss decreases most rapidly (the gradient), and step in that direction.
- **Repeat** this process, updating $w$ and $b$, until we can't get much better—convergence!

Even though computers usually solve linear regression with one matrix formula (the "normal equation"), gradient descent is fundamental. It's the backbone of how we train neural networks!

---

## Demonstration: Linear Regression in PyTorch

Let’s put theory into practice. We'll generate some fake data and fit a single-feature linear regression model using PyTorch—and we'll actually **manually implement the loss and update rules**, just like the math above!

### Installation

First, if you need to install PyTorch and matplotlib, use:

```sh
uv pip install torch matplotlib
```

### Step 1: Create Fake Data

```python
import torch
import matplotlib.pyplot as plt

# Make reproducible
torch.manual_seed(42)

true_w = 2.0    # True slope
true_b = -1.0   # True intercept

N = 100  # Number of data points

# x: Uniformly spaced between -1 and 1
x: torch.Tensor = torch.linspace(-1, 1, N).unsqueeze(1)  # Shape (N, 1)
# y: Linear relation plus some noise
y: torch.Tensor = true_w * x + true_b + 0.1 * torch.randn(N, 1)

plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fake Linear Data')
plt.legend()
plt.show()
```

---

### Step 2: Hand-Coded Linear Regression Training Loop

```python
# Initialize parameters (weights and bias) randomly
w: torch.Tensor = torch.randn(1, requires_grad=False)
b: torch.Tensor = torch.randn(1, requires_grad=False)

learning_rate: float = 0.1
num_epochs: int = 50

losses = []

for epoch in range(num_epochs):
    # 1. Forward pass: compute predictions
    y_pred: torch.Tensor = w * x + b

    # 2. Compute loss (mean squared error)
    loss: torch.Tensor = ((y_pred - y) ** 2).mean()
    losses.append(loss.item())

    # 3. Compute gradients (dL/dw and dL/db) using math above
    grad_w: torch.Tensor = (2 / N) * torch.sum((y_pred - y) * x)
    grad_b: torch.Tensor = (2 / N) * torch.sum(y_pred - y)

    # 4. Update parameters manually
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\nLearned parameters: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"True parameters:    w = {true_w:.4f}, b = {true_b:.4f}")
```

---

### Step 3: Visualize Results

```python
# Plot data and learned regression line
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), (w * x + b).detach().numpy(), color='red', label='Learned Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit (PyTorch)')
plt.show()

# Plot loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss During Training')
plt.show()
```

You should see the line gradually fit the data, and the loss drop smoothly.

---

## Exercises

### Exercise 1: Try a Different Learning Rate

Try setting `learning_rate = 1.0` and `learning_rate = 0.01` in the training loop above.  
- What happens to the convergence and the final loss?
- Why?

**Solution:**  
```python
# Just change 'learning_rate' to 1.0 or 0.01 in the above loop and observe results.
```
- **High learning rate (1.0):** The loss may "explode" or oscillate, never converging.
- **Low learning rate (0.01):** The loss drops very slowly; convergence is much slower.

---

### Exercise 2: Add a Second Feature

Instead of $y = w x + b$, generate new data with $y = w_1 x_1 + w_2 x_2 + b$ for random $x_1, x_2$.  
- Update your code to fit both weights $w_1$, $w_2$, and $b$.

**Solution:**
```python
# Generate data with two features
torch.manual_seed(42)
true_w = torch.tensor([1.5, -3.0])  # True weights
true_b = 0.5
N = 100

X: torch.Tensor = torch.rand(N, 2) * 2 - 1  # random values in [-1, 1]
y: torch.Tensor = X @ true_w + true_b + 0.1 * torch.randn(N)

# Initialize weights
w: torch.Tensor = torch.randn(2, requires_grad=False)
b: torch.Tensor = torch.randn(1, requires_grad=False)

learning_rate = 0.1
for epoch in range(50):
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()

    grad_w = (2 / N) * (X * (y_pred - y).unsqueeze(1)).sum(dim=0)
    grad_b = (2 / N) * (y_pred - y).sum()

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print(f"Learned w: {w}")
print(f"True w:    {true_w}")
```

---

### Exercise 3: Replace Manual Gradients with PyTorch’s Autograd

Try rewriting the training loop using PyTorch’s auto-differentiation, so you don’t hand-code the gradients.

**Solution:**
```python
# Parameters: set requires_grad=True
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.1)

for epoch in range(50):
    optimizer.zero_grad()
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
```
- PyTorch computes gradients for you with `.backward()`!

---

## Summary

We've seen how **linear regression** works:
- It's about fitting the "best" straight line (or hyperplane) to predict $y$ from $x$.
- The model is $y = wx + b$.
- We measure our fit using **mean squared error**.
- We minimize loss by tweaking $w$ and $b$ using **gradient descent** updates derived from calculus.
- Even with more complex models, the same ideas apply!

---

## Key Takeaways

- **Linear regression** is the simplest, clearest example of a machine learning model and training loop.
- **MSE Loss** gives us a clear, quantitative measure of model error.
- **Gradient descent** is a powerful, general-purpose method for learning model parameters—from linear regression up to deep neural networks.
- PyTorch lets us both hand-code and automate gradients, which is at the heart of modern ML frameworks.

Next time, we’ll dig deeper into the details of how we compute gradients—by hand and with PyTorch’s “autograd”—and see why this matters for more powerful models!

---

**USER:**

there are two weird things that happens in the first exercise. when i lower the learning rate to 0.01, it no longer fits. and when i raise it to 1.0, the entire line "fits", but offset, like the weights changed, but the bias is wrong. are these results to be expected?

**ASSISTANT:**

Great observations! Let’s analyze each case in the context of the **manual gradient descent linear regression** code provided.

---

## 1. **Very Low Learning Rate (0.01): Model Fails to Fit Completely**
### **What you observed:**
- The line doesn’t quite fit the data, even after 50 steps.

### **Why this happens:**
- **Smaller learning rates mean smaller steps per epoch.**
- With only 50 epochs, parameters don’t get a chance to move enough to minimize the loss.
- With $\text{lr}=0.1$, each step is ten times larger than $\text{lr}=0.01$, so you converge faster.
- **If you increase the number of epochs** (for example, 1000+), you’ll see fitting eventually.
- This is expected — low learning rate = SLOW progress, not a quality problem.

---

## 2. **High Learning Rate (1.0): Line "Fits" But Offset — Weights Good, Bias Bad**
### **What you observed:**
- The learned **slope** is close to the true value, but the **bias** (intercept) is way off.
- The result is a line that is parallel to the ideal one, but vertically shifted.

### **Why this happens:**
- With such a large learning rate, the updates for $w$ and $b$ both become large.
- **However, $w$ and $b$ can end up being updated at very different *rates***—especially since $x$ is in $[-1, 1]$ and $b$ aggregates all offsets.
- Sometimes, with aggressive stepping, weights can happen to "align" with the correct slope, but bias oscillates or diverges, since its gradient is the sum over all $\hat{y} - y$ (and there's no $x$ to dampen/smooth it).
- Overshooting or imbalance between the $w$ and $b$ updates leads to this parallel/offset behavior.

**This is a common phenomenon with large learning rates** — especially in simple hand-coded loops, where we don’t use tricks like momentum or learning rate schedules.

---

## **How to Check and Improve**

- Set `learning_rate=0.1` — the fitting should be reasonably fast and accurate.
- Lower learning rate ($0.01$): **Increase epochs** (try 1000); you’ll see it fit eventually.
- Higher learning rate ($1.0$): Usually, both $w$ and $b$ diverge or oscillate. If one fits and the other doesn’t, that’s a hint at update imbalance or instability.

**Advanced Note:**  
Popular optimizers like **Adam** or learning rate schedules dampen these effects, but in our hand-coded example, it’s much more visible.

---

## **Bottom Line**

- **Yes, these results are expected** with this manual update, especially in low/high learning rate extremes.
- **Learning rate is a CRITICAL hyperparameter**, and finding a value that’s “just right” is usually needed for stable, fast convergence.
- For slow progress: train longer; for oscillation: lower the step size.

---

### **Try This: Plot the Training Curves**

Plot `w` and `b` over time for different learning rates — you'll see that with too high a learning rate, $b$ might jump around or diverge, while with too low, both update slowly.

---

If you want, I can provide example code to visualize how $w$ and $b$ change over training for different learning rates!

**USER:**

i tried doing exercise 3 with the autograd, but it doesn't work. here is the code:

```python
import torch
import matplotlib.pyplot as plt
from typing import List

# Make reproducible
torch.manual_seed(42)

true_w = 2.0  # True slope
true_b = -1.0  # True intercept

N = 100  # Number of data points

# x: Uniformly spaced between -1 and 1
x: torch.Tensor = torch.linspace(-1, 1, N).unsqueeze(1)  # Shape (N, 1)
# y: Linear relation plus some noise
y: torch.Tensor = true_w * x + true_b + 0.1 * torch.randn(N, 1)

plt.scatter(x.numpy(), y.numpy(), label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fake Linear Data")
plt.legend()
plt.show()

# Initialize parameters (weights and bias) randomly
w: torch.Tensor = torch.randn(1, requires_grad=False)
b: torch.Tensor = torch.randn(1, requires_grad=False)

learning_rate: float = 0.1
num_epochs: int = 50

losses: List[float] = []

optimizer = torch.optim.SGD([w, b], lr=0.1)

for epoch in range(50):
    optimizer.zero_grad()
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:2d}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}"
        )

print(f"\nLearned parameters: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"True parameters:    w = {true_w:.4f}, b = {true_b:.4f}")

# Plot data and learned regression line
plt.scatter(x.numpy(), y.numpy(), label="Data")
plt.plot(x.numpy(), (w * x + b).detach().numpy(), color="red", label="Learned Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit (PyTorch)")
plt.show()

# Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss During Training")
plt.show()
```

here is the error:

```
Traceback (most recent call last):
  File "/Users/ryan/dev/artintellica/py/blog-0097-training-loop-12/ex_3.py", line 40, in <module>
    loss.backward()
  File "/Users/ryan/dev/artintellica/.venv/lib/python3.11/site-packages/torch/_tensor.py", line 648, in backward
    torch.autograd.backward(
  File "/Users/ryan/dev/artintellica/.venv/lib/python3.11/site-packages/torch/autograd/__init__.py", line 353, in backward
    _engine_run_backward(
  File "/Users/ryan/dev/artintellica/.venv/lib/python3.11/site-packages/torch/autograd/graph.py", line 824, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

did i do something wrong? or can this be fixed?
