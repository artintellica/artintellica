+++
title = "Learn the Training Loop with PyTorch, Part 3.2: Learning Rate Scheduling"
author = "Artintellica"
date = "2025-06-18"
model = "o3"
userDelimiter = "**USER:**"
assistantDelimiter = "**ASSISTANT:**"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0110-training-loop-32"
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

that's it, now can you please write blog post 3.2?

the description is:

**2. Learning Rate Scheduling**
- How/why learning rates are changed during training
- Demo: Effect of learning rate on convergence and results

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## **Module 3, Section 2: Learning Rate Scheduling**

---

### **Introduction: Why Adjust the Learning Rate?**

When training a neural network, the "learning rate" is a critical value that controls how much we update the network's weights at each step. If the learning rate is too **high**, our model might never settle down and miss the best solution. If it’s too **low**, the training could take forever or get stuck in a bad spot.

But here's the catch: **the best learning rate might change during training**! That’s where **learning rate scheduling** comes in. We can start training with a higher learning rate (for big, fast steps), then lower it later (for smaller, fine-tuning steps) to help the model settle into the best possible solution.

---

### **ELI5: Learning Rate Scheduling Explained**

Imagine you're trying to find the lowest point in a foggy valley:

- If you take *huge* leaps, you might bounce past the bottom, never settling down.
- If you tiptoe really slowly the entire way, it takes ages to reach the bottom.
- The smartest way? Take big steps at first to get close quickly, then smaller and smaller steps as you zero in on the actual lowest point.

**Learning rate scheduling** is the machine learning version of gradually shrinking those steps, helping us train smarter and faster.

---

### **Mathematical Foundations**

Let’s dive into the math.

A neural network learns by updating its parameters $\theta$ (these are its weights and biases) at each step to minimize some loss function $L(\theta)$. The vanilla update step looks like:

$$
\theta_{t+1} = \theta_{t} - \eta \cdot \nabla_\theta L(\theta_t)
$$

- $\theta_{t}$: parameters at step $t$
- $\eta$: learning rate
- $\nabla_\theta L(\theta_t)$: gradient of the loss function with respect to the parameters

With **learning rate scheduling**, we change $\eta$ over time:

$$
\theta_{t+1} = \theta_{t} - \eta_{t} \cdot \nabla_\theta L(\theta_t)
$$

There are several popular ways to schedule $\eta_t$:
- **Step Decay:** Reduce the learning rate by a factor every _k_ steps.
  
  $$
  \eta_t = \eta_0 \cdot \gamma^{\left\lfloor \frac{t}{k} \right\rfloor}
  $$
  where $\gamma$ is the decay rate (e.g. 0.1).
- **Exponential Decay:** Decrease the learning rate exponentially with time.

  $$
  \eta_t = \eta_0 \cdot \exp(-\lambda t)
  $$
- **Plateau/Performance-based:** Lower the learning rate when the validation loss stops improving.

---

### **Detailed Explanation**

#### **Why Use Schedulers?**

- **Faster Convergence:** Start with large steps to reach near the optimum quickly.
- **Stable Fine-tuning:** Gradually reduce step size to avoid overshooting or oscillating when close to the best solution.
- **Escape Plateaus:** Lowering the learning rate can help the optimizer break out of flat regions or shallow local minima.

#### **PyTorch's Take**

PyTorch makes implementing learning rate schedules extremely easy with `torch.optim.lr_scheduler`. You wrap your optimizer with a scheduler, and step the scheduler every epoch.

Some useful built-in schedulers:

- [`StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html): Reduces LR by a factor every few epochs.
- [`ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html): Exponentially decays LR.
- [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html): Reduces LR based on validation performance.

---

### **Demonstration: The Effect of Learning Rate Scheduling**

Let’s see learning rate scheduling in action! We'll train a simple neural network on the synthetic 2D dataset and compare training with:

1. Constant learning rate
2. Step decay learning rate

#### **Installation Instructions**

If you haven't already, install the necessary libraries:

```bash
uv pip install torch matplotlib
```

#### **Code Example: Training with and without a Scheduler**

```python
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)

# Generate a simple synthetic dataset: y = 2x + 1 + noise
def make_data(n_samples: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(-1, 1, n_samples).unsqueeze(1)
    y = 2 * X + 1 + 0.2 * torch.randn_like(X)
    return X, y

X, y = make_data()

# Define a simple one-layer neural net (just linear regression)
class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# Training loop template
def train(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    lr: float, 
    epochs: int, 
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses

# Settings
epochs = 100
lr = 0.1

# Training with constant learning rate
model_a = SimpleNet()
losses_const = train(model_a, X, y, lr=lr, epochs=epochs)

# Training with step LR scheduler (decay by 0.1 at epoch 50)
model_b = SimpleNet()
optimizer_b = SGD(model_b.parameters(), lr=lr)
scheduler_b = StepLR(optimizer_b, step_size=50, gamma=0.1)
def train_with_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int
) -> list[float]:
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses

losses_sched = train_with_optimizer(model_b, optimizer_b, scheduler_b, X, y, epochs)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_sched, label="Step Decay LR (0.1 → 0.01 at 50)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Effect of Learning Rate Scheduling")
plt.legend()
plt.grid(True)
plt.show()
```

#### **What Happens?**

You should see that:

- **Constant LR:** Training is fast at first, but the loss might "hover" and not get as low, or can oscillate.
- **Scheduled LR:** Drops quickly at first, then after the learning rate decays, the loss keeps decreasing and stabilizes better.

This simple example shows how a good scheduler can help your model reach a better solution!

---

### **Exercises**

#### **Exercise 1: Try Exponential Decay Scheduler**

Train a model using `torch.optim.lr_scheduler.ExponentialLR` and compare its loss curve to the two approaches above. Use a `gamma` value of 0.95.

```python
# Exercise 1 Solution

model_c = SimpleNet()
optimizer_c = SGD(model_c.parameters(), lr=lr)
scheduler_c = torch.optim.lr_scheduler.ExponentialLR(optimizer_c, gamma=0.95)

def train_exp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int
) -> list[float]:
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses

losses_exp = train_exp(model_c, optimizer_c, scheduler_c, X, y, epochs)

plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_sched, label="Step Decay LR")
plt.plot(losses_exp, label="Exponential Decay LR ($\gamma=0.95$)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Comparing Learning Rate Schedules")
plt.legend()
plt.grid(True)
plt.show()
```

---

#### **Exercise 2: Performance-Based Scheduler**

Use `ReduceLROnPlateau` to drop the learning rate when the loss isn't improving. Set `patience=10`, `factor=0.5`.

```python
# Exercise 2 Solution

model_d = SimpleNet()
optimizer_d = SGD(model_d.parameters(), lr=lr)
scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, patience=10, factor=0.5, verbose=True)

criterion = nn.MSELoss()
losses_plateau = []
for epoch in range(epochs):
    model_d.train()
    optimizer_d.zero_grad()
    out = model_d(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer_d.step()
    losses_plateau.append(loss.item())
    scheduler_d.step(loss.item())

plt.figure(figsize=(8,6))
plt.plot(losses_const, label="Constant LR (0.1)")
plt.plot(losses_plateau, label="ReduceLROnPlateau")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Effect of ReduceLROnPlateau")
plt.legend()
plt.grid(True)
plt.show()
```

---

### **Summary**

**Learning rate scheduling** is a simple but powerful tool for optimizing how your model learns. By dynamically adjusting the step size over the course of training, you can achieve both **faster convergence** and **better accuracy**.

---

### **Key Takeaways**

- The **learning rate** is one of the most important hyperparameters for training neural networks.
- **Schedulers** allow us to lower the learning rate as training progresses, which improves optimization and final performance.
- PyTorch provides several built-in schedulers, including **StepLR**, **ExponentialLR**, and **ReduceLROnPlateau**.
- Always **experiment with learning rate schedules**: it can make the difference between a mediocre and a great model!

---

Stay tuned for the next lesson, where we dive into **weight initialization** and why your training loop can benefit from choosing smarter starting points!

**USER:**


