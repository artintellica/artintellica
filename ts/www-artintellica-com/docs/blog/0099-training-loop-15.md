+++
title = "Learn the Training Loop with PyTorch, Part 1.5: Numerical vs. Analytical Gradients"
author = "Artintellica"
date = "2025-06-16"
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
- **Accompany every lesson with practical code examples.**
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

that's it, now can you please write blog post 1.5? the description is:

**5. Numerical vs. Analytical Gradients**
   - How are gradients computed?  
   - Manual calculation and confirmation with numerical gradient

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 1.5: Numerical vs. Analytical Gradients

### Introduction

In the previous lessons, we learned what gradients are and why they're essential for training models using methods like gradient descent. But **how** are these gradients actually calculated? In this lesson, we'll demystify two main ways to compute gradients: **analytically** (using calculus rules) and **numerically** (using finite differences). Understanding both is extremely important—manual checks with numerical gradients can catch bugs and help you build confidence in your implementation.

---

## ELI5: What Are Gradients and How Can We Compute Them?

Imagine you’re on a hike and want to descend a hill as quickly as possible. The **gradient** at your feet tells you which direction is the steepest way down. But you don’t have a map! 

- **Analytical gradients:** This is like being given an exact formula for the hill so you always know which way is steepest.
- **Numerical gradients:** This is like moving a tiny bit in each direction, measuring if you’ve gone up or down, and deducing which way is down.

Analytical is fast and exact (if you do the math right), while numerical is foolproof but a bit slow and less precise. Checking analytical gradients with numerical ones is like double-checking your directions on the trail using both a map and a compass.

---

## Mathematical Foundations

### Analytical Gradients

Suppose you have a function $f(x)$ and you want to know *how* $f$ changes when $x$ changes. The **gradient** is simply the derivative: $\frac{df}{dx}$.

**Example:** For $f(x) = x^2$,

$$
\frac{df}{dx} = 2x
$$

In machine learning, our "function" is usually the **loss** as a function of parameters (weights), and we want $\nabla_{\mathbf{w}} L(\mathbf{w})$ — the gradient of loss with respect to weights.

- Analytical gradients are calculated using calculus—chain rule, product rule, etc.

### Numerical Gradients

Numerical gradients use *finite differences*: we make a tiny change and see how the output changes.

For a function $f(x)$, the numerical gradient at $x$ is approximated as:

$$
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
$$

where $h$ is a very small number (e.g., $1e^{-5}$).

- This "symmetric difference quotient" gives a good estimate of the slope of $f$ at $x$.

#### Why Compare Numerically and Analytically?

- **Numerical gradients** give you a way to check if your analytical/automatic gradients (e.g., via PyTorch) are correct.
- Useful for bug-hunting and for building trust in your implementation.

---

## In-Depth Explanation

Gradient calculation is at the heart of neural network training. In frameworks like PyTorch, gradients are typically computed **automatically** using something called *autograd* (automatic differentiation).

But what if you want to check if those gradients are right? Or you just want to build intuition and test your code? Here’s where numerical gradients are invaluable. They are computed by slightly nudging each parameter, one at a time, and observing how much the loss changes. This is reliable—but slow—since it requires re-evaluating the function many times.

Let's recap the two approaches:

- **Analytical (Automatic) Gradients:** Computed mathematically using calculus, often by backpropagation and the chain rule, and done instantly by libraries like PyTorch.
- **Numerical Gradients:** Computed by re-running the model with slightly perturbed parameters to estimate the slope. Not as fast, but almost guaranteed to be correct if you’re careful.

### The Recipe:

1. **Pick a tiny value for $h$** (like $1e^{-5}$).
2. For each parameter:
    1. Slightly increase by $h$: $x_i + h$.
    2. Slightly decrease by $h$: $x_i - h$.
    3. Compute loss at both locations.
    4. Use the formula to estimate the derivative.
3. Compare to the gradient calculated analytically (by PyTorch).

---

## Demonstration: Gradient Checking for Linear Regression in PyTorch

Let's put this into practice with PyTorch!

### **Setup**

We will:
- Define a simple 1D linear regression model.
- Compute the loss and its gradient (analytically, via PyTorch autograd).
- Compute the loss and gradient numerically for the weights.
- Compare the two.

#### **Install Dependencies**

```bash
uv pip install torch matplotlib
```

#### **Code: Analytical vs. Numerical Gradients**

```python
import torch
import matplotlib.pyplot as plt
from typing import Tuple

def simple_linear_regression_data(n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # y = 2x + 3 + noise
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * X + 3 + 0.1 * torch.randn(n, 1)
    return X, y

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean()

def model(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ w + b

def numerical_gradient(
    func, 
    params: Tuple[torch.Tensor, ...], 
    param_idx: int, 
    h: float = 1e-5
) -> torch.Tensor:
    grads = torch.zeros_like(params[param_idx])
    it = torch.nditer(params[param_idx], flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = params[param_idx][idx].item()

        params_plus = [p.clone() for p in params]
        params_minus = [p.clone() for p in params]
        params_plus[param_idx][idx] = orig + h
        params_minus[param_idx][idx] = orig - h
        f_plus = func(*params_plus).item()
        f_minus = func(*params_minus).item()
        grad = (f_plus - f_minus) / (2 * h)
        grads[idx] = grad

        it.iternext()
    return grads

# Generate data
X, y = simple_linear_regression_data(20)
# Initialize parameters with grad enabled
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def loss_fn(w_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
    y_pred = model(X, w_, b_)
    return mse_loss(y_pred, y)

# Compute analytical gradients
loss = loss_fn(w, b)
loss.backward()  # This fills in w.grad and b.grad

# Compute numerical gradients
num_grad_w = numerical_gradient(loss_fn, (w, b), param_idx=0)
num_grad_b = numerical_gradient(loss_fn, (w, b), param_idx=1)

print("Analytical grad (w):", w.grad)
print("Numerical grad (w): ", num_grad_w)
print("Analytical grad (b):", b.grad)
print("Numerical grad (b): ", num_grad_b)
```

### **Analysis: Are They Close?**

The two gradients should be very close. If you see a big difference, there’s likely a bug somewhere—either in your code or in the choice of $h$ (if it’s too big or too small, numerical gradients get inaccurate).

---

## Exercises

### 1. **Change the Model to Quadratic and Compare Gradients**

- Modify the model to $y = w_2 x^2 + w_1 x + b$. Compute and compare gradients for $w_1$, $w_2$, and $b$.

```python
# Quadratic model: y = w2 * x^2 + w1 * x + b

def quadratic_model(X: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return w2 * X ** 2 + w1 * X + b

# New parameters
w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def loss_fn_quad(w1_: torch.Tensor, w2_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
    y_pred = quadratic_model(X, w1_, w2_, b_)
    return mse_loss(y_pred, y)

loss_quad = loss_fn_quad(w1, w2, b)
loss_quad.backward()

num_grad_w1 = numerical_gradient(loss_fn_quad, (w1, w2, b), param_idx=0)
num_grad_w2 = numerical_gradient(loss_fn_quad, (w1, w2, b), param_idx=1)
num_grad_b_quad = numerical_gradient(loss_fn_quad, (w1, w2, b), param_idx=2)

print("Analytical grad (w1):", w1.grad)
print("Numerical grad (w1):", num_grad_w1)
print("Analytical grad (w2):", w2.grad)
print("Numerical grad (w2):", num_grad_w2)
print("Analytical grad (b):", b.grad)
print("Numerical grad (b):", num_grad_b_quad)
```

---

### 2. **Visualize the Loss as a Function of the Weight**

- Pick a fixed $b$, sweep $w$ from -4 to 4, and plot the loss.

```python
import numpy as np

b_fixed = b.detach().item()
ws = np.linspace(-4, 4, 100)
losses = []
for w_test in ws:
    w_tensor = torch.tensor([[w_test]])
    l = loss_fn(w_tensor, torch.tensor([b_fixed]))
    losses.append(l.item())

plt.figure()
plt.plot(ws, losses, label='Loss vs w')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Loss as a function of w')
plt.legend()
plt.show()
```

---

### 3. **Try Different $h$ Values for Numerical Gradient**

- Try $h = 1e^{-3}$ and $h = 1e^{-7}$ and observe how the numerical gradient changes. What happens if $h$ is too big or too small?

```python
h_list = [1e-3, 1e-5, 1e-7]
for h in h_list:
    num_grad_w = numerical_gradient(loss_fn, (w, b), param_idx=0, h=h)
    print(f"Numerical grad (w) with h={h:.0e}:", num_grad_w)
```

---

## Summary

In this post, we explored the two major ways to compute gradients for loss functions: **analytical gradients** (using calculus or autograd in PyTorch) and **numerical gradients** (using the finite difference method). 

- Analytical gradients are fast and exact (as long as your math or library is correct).
- Numerical gradients are slower, but provide an invaluable way to check your work, especially in debugging or during implementation from scratch.

**Key Takeaways:**

- Gradients are central to training—computing them correctly is crucial!
- Analytical (autograd) and numerical gradients should match closely for well-behaved functions.
- Numerical gradient checking can catch subtle bugs in your code.
- For modern machine learning, autograd (analytical) is always used in practice, but gradient checking is a vital skill—especially when experimenting or learning.

**Next up:** We'll summarize Module 1 and set up for nonlinear models and neural networks. See you in the next lesson!

**USER:**


