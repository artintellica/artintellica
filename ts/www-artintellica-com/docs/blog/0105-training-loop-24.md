+++
title = "Learn the Training Loop with PyTorch, Part 2.4: The Role of Activations"
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
- then provide the mathematical foundations. for math, always use dollar delimited LaTeX, and always provide a brief explanation of the math. for instance, $\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}$ is the equation for linear regression, where $\mathbf{y}$ is the output vector, $\mathbf{X}$ is the input matrix, $\mathbf{w}$ is the weight vector, and $\mathbf{b}$ is the bias vector.
- for block-level math, always use two dollar signs, like this:
  $$
  \mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
  $$
- after the mathematical foundations, such as definitions, then always provide an in-depth explanation, covering the topic in detail, and expanding on the explanation of the math if necessary.
- for other formatting, use markdown like headings, bullet points, and code blocks to make the content readable. do not use HTML tags.
- after the math is complete, then provide one or more demonstrations of the topic using PyTorch. if possible, always use PyTorch, not numpy, just to keep it simple and consistent. you can also use matplotlib to visualize the results.
- for all python code, always use type hints, and, if necessary, describe installation instructions using "uv" (not pip).
- then, create a series of exercises, somewhere from 1 - 4 total exercises that cover the material. please provide the soruce code for each exercise to make sure the student can easily compare their solution to the correct answer. do not hide the exercises and do not put HTML in the exercises.
- finally, provide a summary of the topic, and a list of key takeaways.

that's it, now can you please write blog post 2.4?

the description is:

**4. The Role of Activations**
   - Intuitive explanation and code examples
   - Effect on learning and expressiveness

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 2.4: The Role of Activations

---

### Introduction

Welcome back to "Learn the Training Loop with PyTorch"!  
In the last blog post, you built a neural network from scratch and learned how it could learn more complex relationships than linear regression. But you might be wondering: *what really makes a neural network “non-linear,” and why is that so important*?  

Today, we’re zooming in on **activation functions**—the key ingredient that makes neural networks more expressive and capable than linear models. We'll intuitively explain what activations are, why they're needed, look at some of the most common ones, see their math, and finally visualize their impact with code.

---

## ELI5: What Is an Activation Function?  

Think of a neural network like an assembly line. At each step (layer), worker robots (neurons) add their ingredient (multiply input by weights and add bias), then pass it to the next worker. If each worker only ever demanded the *exact* recipe (just multiplying and adding), the final product would *always* be like the ingredients mixed in the same boring way, no matter how many steps you have.

An **activation function** is like a taste-tester in between workers. Each time, the taste-tester decides whether and how much of the new ingredient should go on, sometimes amplifying, squashing, or cutting things off entirely!  

Without an activation function, no matter how many layers you add, the network can only make *linear* changes to the input—like a single-layer recipe, just longer and more expensive. Activation functions let your network "think outside the line"—mixing, squashing, and transforming signals so it can learn curves, corners, and real-life complexities.

---

## Mathematical Foundations

Let’s formalize where activation functions fit into the neural network workflow.

A *single neuron* computes:

$$
a = f(z)
$$

where

- $z = \mathbf{w}^\top \mathbf{x} + b$  
   - $\mathbf{x}$: input vector  
   - $\mathbf{w}$: weight vector  
   - $b$: bias  
- $f(\cdot)$: **activation function** (e.g., ReLU, sigmoid, tanh)

A *layer* of neurons vectorizes this:

$$
\mathbf{a} = f(\mathbf{W} \mathbf{x} + \mathbf{b})
$$

where

- $\mathbf{W}$: weight matrix  
- $\mathbf{b}$: bias vector  
- $f$ is applied *element-wise* to the linear transformation's output.

Some **common activation functions**:

- **Sigmoid:**  
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
- **Tanh:**  
  $$
  \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
  $$
- **ReLU (Rectified Linear Unit):**  
  $$
  \mathrm{ReLU}(z) = \max(0, z)
  $$

**Why does it matter?**  
If you "stack" linear layers *without* a non-linearity between them, the result is still a linear function of the input—no more powerful than a single layer! Activations unlock *non-linear* transformations, giving networks the capacity to model almost any function.

---

## Deep Dive: Why Activation Functions Matter

Let’s break it down:

1. **Linear Layers Without Activation = Linear Model**  
    Suppose you have two linear layers:
    $$
    \mathbf{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
    $$
    You can rewrite this as:
    $$
    \mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{c}
    $$
    where $\mathbf{A} = \mathbf{W}_2\mathbf{W}_1$, $\mathbf{c} = \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2$  
    -- still just a single linear transformation! So, *no matter how many you stack, without activations they're just one big linear operation.*

2. **Introducing Non-Linearity**
    When you wrap a non-linear activation $f(\cdot)$ between the layers:
    $$
    \mathbf{y} = \mathbf{W}_2 \cdot f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
    $$
    Now, the *relationship* between $\mathbf{x}$ and $\mathbf{y}$ can curve, bend, or even oscillate—depending on $f$. This makes neural networks *universal function approximators*.

3. **Practical Impacts**
    - Networks with activations can solve complex tasks (like language translation, image recognition, etc.).
    - Each activation function has unique properties. For example:
        - **Sigmoid:** outputs between $0$ and $1$, useful for probabilities, but can “saturate” and slow down learning.
        - **Tanh:** similar to sigmoid but outputs between $-1$ and $1$ (centered around $0$).
        - **ReLU:** super simple, fast to compute, and the default in modern networks—but "dies" (outputs 0) for negative values.

4. **Choosing the Right Activation**
    - Standard hidden layers: **ReLU** (or its variants)
    - Outputs for probabilities: **Sigmoid** (binary) or **Softmax** (multiclass)
    - Outputs for regression: use *no* activation (linear output) or sometimes **Tanh** for bounded outputs

---

## Demonstrations in PyTorch

Let’s visualize activation functions and see their *impact* on network expressiveness!

> **Install requirements (if needed):**
> ```
> uv pip install torch matplotlib
> ```

### 1. Visualizing Common Activation Functions

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_activations() -> None:
    z = torch.linspace(-5, 5, steps=200)

    activations = {
        "ReLU": F.relu(z),
        "Sigmoid": torch.sigmoid(z),
        "Tanh": torch.tanh(z),
    }

    plt.figure(figsize=(8, 5))
    for name, activation in activations.items():
        plt.plot(z.numpy(), activation.numpy(), label=name)
    plt.title("Common Activation Functions")
    plt.xlabel("Input z")
    plt.ylabel("Activation(z)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_activations()
```

### 2. Why We Need Non-linear Activations: Model Expressiveness

Let’s fit a toy dataset with and without non-linearity.

```python
import torch
import matplotlib.pyplot as plt

def make_toy_data(n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    # Data: input x, labels y = sin(x)
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n).unsqueeze(1)
    y = torch.sin(x)
    return x, y

class SimpleNet(torch.nn.Module):
    def __init__(self, use_activation: bool):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.use_activation = use_activation
        self.fc2 = torch.nn.Linear(10, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_activation:
            x = F.relu(x)  # Try also with torch.tanh(x)
        x = self.fc2(x)
        return x

def train_model(use_activation: bool, epochs: int = 1000) -> torch.nn.Module:
    x, y = make_toy_data()
    model = SimpleNet(use_activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def plot_fit_with_and_without_activation():
    x, y = make_toy_data()
    models = {
        "No Activation (Linear Only)": train_model(use_activation=False, epochs=1000),
        "With ReLU Activation": train_model(use_activation=True, epochs=1000),
    }

    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy(), label="True sin(x)", color="black", linewidth=2)
    for label, model in models.items():
        y_pred = model(x).detach()
        plt.plot(x.numpy(), y_pred.numpy(), label=label)

    plt.title("Linear v.s. Non-linear (ReLU) Neural Network Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_fit_with_and_without_activation()
```

Try replacing `F.relu(x)` with `torch.tanh(x)` in the model above, and see how much better the fit gets for a sinusoidal function!

---

## Exercises

### 1. **Exercise: Plot the Derivative of Each Activation**

Modify the given activation plotting code to plot not just each function, but its derivative as well.  
*Why?* The gradient affects learning speed and stability!

```python
def plot_activation_and_derivative() -> None:
    z = torch.linspace(-5, 5, steps=200)
    activations = {
        "ReLU": (F.relu(z), (z > 0).float()),
        "Sigmoid": (torch.sigmoid(z), torch.sigmoid(z) * (1 - torch.sigmoid(z))),
        "Tanh": (torch.tanh(z), 1 - torch.tanh(z) ** 2),
    }
    plt.figure(figsize=(10, 6))
    for idx, (name, (val, grad)) in enumerate(activations.items()):
        plt.subplot(2, 3, idx+1)
        plt.plot(z.numpy(), val.numpy())
        plt.title(f"{name}")
        plt.subplot(2, 3, idx+4)
        plt.plot(z.numpy(), grad.numpy())
        plt.title(f"{name} Derivative")
    plt.tight_layout()
    plt.show()

plot_activation_and_derivative()
```

---

### 2. **Exercise: Swap in Tanh for ReLU**

In the toy function-fitting code above, swap the `F.relu(x)` to `torch.tanh(x)` and retrain. Which activation works better for approximating $\sin(x)$?  
*Answer: Tanh matches the shape much better, because its output range $(-1, 1)$ matches $\sin(x)$'s output!*

---

### 3. **Exercise: Build a Custom Activation Function**

Create a custom PyTorch activation function called "Swish," defined as $f(z) = z \cdot \sigma(z)$.

```python
def swish(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)

# Visualize Swish
z = torch.linspace(-5, 5, steps=200)
plt.plot(z.numpy(), swish(z).numpy(), label="Swish")
plt.legend()
plt.grid(True)
plt.title("Swish Activation Function")
plt.show()
```

---

## Summary

- **Activation functions** are what make neural networks *nonlinear*—and therefore powerful enough to learn complex, realistic data.
- Stack of *linear* layers (no activation) = *still a linear model!*
- Common activations: **ReLU**, **Sigmoid**, **Tanh**—each with pros and cons. ReLU is simple and most common, while Tanh/Sigmoid are saturating (slow to learn at extremes) but sometimes useful.
- Choice of activation impacts how well your network learns, especially depending on your data and output range.

---

## Key Takeaways

- Activation functions introduce *non-linearity*—the “magic ingredient” in deep learning.
- You must use activations between layers to unlock the power of neural networks.
- Each activation function has unique properties affecting learning and expressiveness.
- Visualization is a great way to understand how activations work!
- *Always* consider your output’s range and the problem type (classification, regression) when choosing activations.

---

**Next up:** Mini-batching and data pipelines—how and why we feed data in small chunks to networks for efficient, robust learning!

**USER:**


