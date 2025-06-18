+++
title = "Learn the Training Loop with PyTorch, Part 3.1: Optimization Algorithms Beyond SGD"
author = "Artintellica"
date = "2025-06-18"
model = "o3"
userDelimiter = "**USER:**"
assistantDelimiter = "**ASSISTANT:**"
+++

**USER:**

i need your help to write the next blog post in a series i am producing called
"Learn the Training Loop with PyTorch".

first, let me give you the outline of the series:

---

## **Module 1: The Elementary Training Loop**

**Goal:** Build foundational intuition by hand-coding a simple training loop
from scratch.

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

**Goal:** Extend the basic loop to more powerful models; introduce neural
network concepts.

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

**Goal:** Connect foundational understanding to the large-scale models used in
industry.

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

- Early stopping, batch/layer normalization, data augmentation (conceptual
  intro)

**8. The Training Loop in Practice: Case Studies**

- High-level overview of how OpenAI/Google/Meta might train LLMs: what’s the
  same, what’s different.
- What "scaling up" really means: dataset, parameters, hardware, tricks.

**9. Conclusion: What's Next After the Training Loop?**

- Discussion: fine-tuning, transfer learning, and unsupervised/self-supervised
  methods.

---

### **General Tips for Each Module:**

- **Start simple, build incrementally:** Keep code simple at first, then show
  how to layer complexity.
- **Accompany every lesson with practical code examples.**
- **Focus on intuition:** Use diagrams, plots, and hands-on demo to make
  concepts tangible.
- **Summarize and review regularly:** Add checklists or quizzes at the end of
  each module.

---

great. now, for each blog post, i am hoping to follow the following structure:

- the course is called "Learn the Training Loop with PyTorch"
- always provide a brief introduction to the topic at hand
- then always provide a very simple "ELI5" explanation of the topic, that
  theortically anyone could understand
- then provide the mathematical foundations. for math, always use dollar
  delimited LaTeX, and always provide a brief explanation of the math. for
  instance, $\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}$ is the equation for
  linear regression, where $\mathbf{y}$ is the output vector, $\mathbf{X}$ is
  the input matrix, $\mathbf{w}$ is the weight vector, and $\mathbf{b}$ is the
  bias vector.
- for block-level math, always use two dollar signs, like this:
  $$
  \mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
  $$
- after the mathematical foundations, such as definitions, then always provide
  an in-depth explanation, covering the topic in detail, and expanding on the
  explanation of the math if necessary.
- for other formatting, use markdown like headings, bullet points, and code
  blocks to make the content readable. do not use HTML tags.
- after the math is complete, then provide one or more demonstrations of the
  topic using PyTorch. if possible, always use PyTorch, not numpy, just to keep
  it simple and consistent. you can also use matplotlib to visualize the
  results.
- for all python code, always use type hints, and, if necessary, describe
  installation instructions using "uv" (not pip).
- then, create a series of exercises, somewhere from 1 - 4 total exercises that
  cover the material. please provide the soruce code for each exercise to make
  sure the student can easily compare their solution to the correct answer. do
  not hide the exercises and do not put HTML in the exercises.
- finally, provide a summary of the topic, and a list of key takeaways.

that's it, now can you please write blog post 3.1?

the description is:

**1. Optimization Algorithms Beyond SGD**

- Adam, RMSProp, etc.: How they work, why they're useful

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch

## Module 3.1: **Optimization Algorithms Beyond SGD**

---

### Introduction

Welcome back to "Learn the Training Loop with PyTorch"! So far, we've trained
models using the most fundamental optimizer: Stochastic Gradient Descent (SGD).
But, have you ever wondered **how modern neural networks actually manage to
train so well and so fast, even when they’re so large and complex?** The secret
lies in improved optimization algorithms.

Today, let's unravel Adam, RMSProp, and their friends. **How do they work,
mathematically and in practice?** And why are they so widely used in deep
learning?

---

### ELI5: "Why Not Just Use SGD?"

Imagine you’re descending a hill blindfolded. SGD is like taking steps directly
downhill, but the size of your steps is always the same. That’s simple, but not
always smart—if the ground is rough or slopes change, you might fumble or even
miss the deepest valley!

Optimizers like **RMSProp** and **Adam** are like having a smart hiking buddy
whisper in your ear:

- "Hey! The path’s rough here, maybe take smaller steps."
- "You’ve been walking in the same direction and keep stumbling, maybe slow
  down..."

**These algorithms adjust your steps—automatically—to help you reach the bottom
faster and more reliably.**

---

### Mathematical Foundations

Let's formalize what an optimizer does:

Suppose we’re minimizing a loss $L(\theta)$ with respect to model parameters
$\theta$. At each step $t$:

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

- $\eta$ is the **learning rate** (step size)
- $g_t = \nabla_\theta L(\theta_t)$ is the gradient at step $t$

#### SGD Recap

Stochastic Gradient Descent keeps $\eta$ fixed and uses only $g_t$ (the current
gradient) to update each parameter.

#### Problem: One Size Doesn’t Fit All

- Some parameters could benefit from bigger/smaller steps.
- Gradients might bounce around or have very different magnitudes, making
  training unstable or slow.

#### RMSProp: Scaling Steps by Recent Gradient Magnitude

RMSProp introduces a **per-parameter learning rate**, adapting it for each
weight depending on past gradient magnitudes.

The RMSProp update is:

$$
\begin{align*}
s_t &= \rho s_{t-1} + (1 - \rho) g_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
\end{align*}
$$

- $s_t$ tracks the _running average_ of past squared gradients.
- $\rho$ is the decay rate (commonly $0.9$).
- $\epsilon$ is a small fudge factor to prevent divide-by-zero (e.g.,
  $1\mathrm{e}{-8}$).
- Now, weights with big/unstable gradients get smaller steps in the update.

#### Adam: First and Second Moments (Momentum + RMSProp)

Adam builds on RMSProp and adds **momentum**, i.e., keeps track of both:

- The "mean" (first moment) of past gradients
- The "variance" (second moment, like RMSProp)

Adam’s update rule:

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align*}
$$

- $m_t$: Running average (momentum) of gradients.
- $v_t$: Running average of squared gradients (like RMSProp).
- $\beta_1, \beta_2$ are decay rates, e.g., $(0.9, 0.999)$
- $\hat{m}_t$, $\hat{v}_t$ are _bias-corrected_ estimates for early steps (to
  ensure they're not too small).

---

### In-Depth Explanation

- **SGD** is the most basic optimizer: it makes updates in the opposite
  direction of the current gradient, using the _same_ step size for every
  parameter, every time. While elegant and actually quite effective, it's
  sensitive to:

  - The scale of gradients (some parameters might need bigger/smaller steps)
  - Noisy updates (especially when gradients vary a lot)
  - Curvature of the loss landscape (slowing down in _valleys_ or _ravines_)

- **RMSProp** solves one of SGD's big headaches: parameters with big, volatile
  gradients can end up taking steps that are too large (causing instability), or
  steps that are too small (slowing learning). By keeping a running history of
  how "noisy" each parameter’s gradient has been, RMSProp divides each step by
  the _recent average_, shrinking updates for bouncy or large-gradient
  parameters.

- **Adam** piles on another trick: momentum. Imagine rolling a ball: momentum
  means you keep some "inertia" from the previous steps, helping you speed up
  when going in a straight line, or smooth out shaky updates. Adam keeps an
  exponentially decaying average of both the mean and variance of gradients, and
  smartly combines them for much more reliable, faster convergence.

- **Why are these so useful?**
  - **Adaptivity:** No need to painstakingly tune the learning rate for each
    weight.
  - **Stability:** Parameters that would “explode” in SGD are automatically
    tamed.
  - **Speed:** Especially on complex loss landscapes, Adam/RMSProp converge
    faster and find good solutions even when SGD gets stuck.

---

### Demonstration: Comparing SGD, RMSProp, and Adam in PyTorch

We'll use a simple regression problem as our playground to see these optimizers
in action.

#### **Installation Requirements**

Make sure you've installed PyTorch and matplotlib (using
[uv](https://github.com/astral-sh/uv)):

```bash
uv pip install torch matplotlib
```

#### **Experiment: Fitting a Linear Model**

We'll fit a simple linear function $y = 2x + 3$ with added noise, using a tiny
neural net (just a single linear layer) and compare optimizers.

```python
import torch
import matplotlib.pyplot as plt
from typing import Tuple

# -- Data Preparation
def make_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    X = torch.linspace(0, 1, n_samples).unsqueeze(1)
    y = 2 * X + 3 + 0.2 * torch.randn_like(X)
    return X, y

X, y = make_data()

# -- Simple Linear Model
class SimpleLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# -- Training function
def train_loop(optimizer_name: str, num_epochs: int = 100) -> Tuple[list[float], SimpleLinear]:
    model = SimpleLinear()
    criterion = torch.nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, alpha=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    else:
        raise ValueError("Unknown optimizer")

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, model

optimizers = ["sgd", "rmsprop", "adam"]
results = {}
for opt in optimizers:
    losses, model = train_loop(opt)
    results[opt] = {"losses": losses, "model": model}

# -- Plotting Loss Curves
plt.figure(figsize=(8,5))
for opt in optimizers:
    plt.plot(results[opt]["losses"], label=opt.upper())
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Convergence of Different Optimizers")
plt.legend()
plt.show()

# -- Plotting Predictions
plt.figure(figsize=(8,5))
plt.scatter(X.numpy(), y.numpy(), label="Data", alpha=0.6)
x_plot = torch.linspace(0, 1, 100).unsqueeze(1)
for opt in optimizers:
    y_pred = results[opt]["model"](x_plot).detach().numpy()
    plt.plot(x_plot.numpy(), y_pred, label=opt.upper())
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Learned Linear Fit by Optimizer")
plt.show()
```

**What do you see?**

- Loss curves: Notice how Adam and RMSProp often converge faster and more
  smoothly than plain SGD.
- Predictions: All optimizers find a good fit, but some do so with fewer epochs
  and less "wiggle" in their learning.

---

### Exercises

#### **1. Tune the Learning Rate**

Change the learning rate for each optimizer in the code above to **0.01**,
**0.05**, and **0.2**. Observe and plot the loss curves for each setting.  
**What happens if your learning rate is too high or too low?**

**Solution:** Replace `lr=0.1` in the optimizer setup with your desired learning
rates. Plotting different curves will reveal the effect.

---

#### **2. Implement Momentum SGD**

PyTorch's `SGD` can use momentum! Modify the optimizer in the training loop to
use momentum (e.g., `momentum=0.9`). Compare its performance to plain SGD.

**Example:**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

Compare the loss curves with and without momentum.

---

#### **3. Try With a Noisier Dataset**

Modify the `make_data` function to have `0.5 * torch.randn_like(X)` instead of
`0.2 * ...`. Which optimizer is most robust to this increased noise?

**Solution:**

```python
def make_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    X = torch.linspace(0, 1, n_samples).unsqueeze(1)
    y = 2 * X + 3 + 0.5 * torch.randn_like(X)
    return X, y
```

Re-run training and compare optimizers.

---

#### **4. Inspect Learned Parameters**

At the end of training, print out the values of model weights and biases for
each optimizer:

```python
for opt in optimizers:
    w = results[opt]["model"].linear.weight.item()
    b = results[opt]["model"].linear.bias.item()
    print(f"{opt.upper()} learns w = {w:.2f}, b = {b:.2f}")
```

**Do they all learn similar parameters? Why/why not?**

---

### Summary

In this post, we explored what makes modern optimizers like **Adam** and
**RMSProp** so powerful compared to simple SGD. Instead of blindly taking steps
based on the current gradient, these optimizers keep a running memory—of how
big, bouncy, or noisy gradients are—and tweak each parameter’s stepsized
accordingly. Adam goes a step further, using momentum and adaptive learning
rates to accelerate and stabilize learning.

---

### Key Takeaways

- **SGD** is simple but can be slow and unstable for complex problems.
- **RMSProp** adapts learning rates per parameter based on the recent magnitude
  of gradients, making training more robust.
- **Adam** combines momentum (smoothing gradients) and per-parameter adaptivity
  for even faster, more reliable convergence.
- PyTorch makes it easy to swap optimizers—experiment with Adam, RMSProp, and
  their settings to improve your models!
- Choosing the right optimizer and learning rates can be the difference between
  a model that _never learns_ and one that _quickly finds a great solution_.

**Up next:** We'll continue exploring powerful training tricks to get the most
out of your neural networks!

**USER:**
