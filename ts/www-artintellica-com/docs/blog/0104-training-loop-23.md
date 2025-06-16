+++
title = "Learn the Training Loop with PyTorch, Part 2.3: Implementing a Simple Neural Net from Scratch"
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

that's it, now can you please write blog post 2.3?

the description is:

**3. Implementing a Simple Neural Net from Scratch**
   - Single hidden layer network in numpy/Python
   - Coding forward, loss, backward, and parameter updates

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 2, Lesson 3: Implementing a Simple Neural Net from Scratch

---

Welcome back to **Learn the Training Loop with PyTorch**! So far, we've understood the basic training loop and dived into the world of neural networks. In this lesson, we're going to get our hands dirty and **build a simple neural network from scratch**—implementing each part ourselves. We'll code a one-hidden-layer "multilayer perceptron" (MLP), covering the forward pass, the loss calculation, the backward pass (i.e., gradients!), and manual parameter updates.

---

## ELI5: What Are We Building, and Why?

Imagine you're building a robot brain that can recognize handwritten numbers. This brain is a neural network—a bunch of little "switches" (neurons) that talk to each other. Each neuron takes some numbers, squishes them together, and passes them on. By chaining a few of these together (layers), our robot can learn all sorts of patterns.

In this lesson, we're building a **baby neural network** with just *one* hidden layer. We'll see, step by step, **how data flows through the network**, **how it learns by looking at its mistakes**, and **how it updates itself to do better next time**.

---

## Mathematical Foundations

Let’s formalize what we mean by a “single hidden layer network.”

### 1. Structure

Suppose our input data is a matrix $\mathbf{X} \in \mathbb{R}^{N \times D_{in}}$, where:
- $N$ = number of data samples (batch size)
- $D_{in}$ = number of input features

We want to map $\mathbf{X}$ to an output $\mathbf{y}_{\text{pred}}$ via a hidden layer:
- $D_{hidden}$ = number of hidden units (width of hidden layer)
- $D_{out}$ = number of output features

The computations are:

$$
\begin{align*}
\text{Hidden layer (linear)}: &\quad \mathbf{h} = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1 \\
\text{Hidden layer (activation)}: &\quad \mathbf{h}_{\text{act}} = \sigma(\mathbf{h}) \\
\text{Output layer:} &\quad \mathbf{y}_{\text{pred}} = \mathbf{h}_{\text{act}} \mathbf{W}_2 + \mathbf{b}_2
\end{align*}
$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{D_{in} \times D_{hidden}}$, $\mathbf{b}_1 \in \mathbb{R}^{D_{hidden}}$
- $\mathbf{W}_2 \in \mathbb{R}^{D_{hidden} \times D_{out}}$, $\mathbf{b}_2 \in \mathbb{R}^{D_{out}}$
- $\sigma(\cdot)$ is an activation function (let’s use ReLU: $\mathrm{ReLU}(x) = \max(0, x)$)

### 2. Loss

For a regression task, we often use **mean squared error** (MSE):

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left\| \mathbf{y}_{\text{pred}, i} - \mathbf{y}_{\text{true}, i} \right\|^2
$$

### 3. Parameter Update (Gradient Descent)

We need to **compute the gradients** of the loss with respect to each parameter ($\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$), then **update in the opposite direction of the gradient**:

$$
\theta \leftarrow \theta - \eta \frac{\partial \text{Loss}}{\partial \theta}
$$

Where $\eta$ is the learning rate.

---

## Step-by-Step Explanation

Let's expand on what's going on:

1. **Initialization**: We start by randomly initializing all weights ($\mathbf{W}_1, \mathbf{W}_2$) and biases ($\mathbf{b}_1, \mathbf{b}_2$).

2. **Forward Pass**: For each batch of inputs $\mathbf{X}$, compute the value at the hidden layer (`h`) via a linear transform, then apply an activation function (ReLU). Take that output, and compute the final output (`y_pred`) via another linear transform.

   - This is called "forward propagation"—the data moves forward through the network.

3. **Compute Loss**: Calculate how far your predictions are from the ground truth, usually via MSE.

4. **Backward Pass (Backpropagation)**: Use the chain rule to calculate the derivatives (gradients) of the loss with respect to every parameter.

   - In code, with PyTorch, this is `loss.backward()`—under the hood, PyTorch builds a computation graph and works everything out for you.

5. **Parameter Update**: For each parameter, adjust it "a little" in the direction that reduces loss, using gradient descent.

6. **Repeat**: Go back to the start! With enough iterations, the model learns patterns in the data.

---

## Implementing the Neural Network in PyTorch

Let's put this all into practice by building a simple one-hidden-layer neural network with **manual parameter updates** (no nn.Module, no torch.optim!).

We'll do a basic regression: the network will learn to fit a simple function.

### Installation

If you haven’t installed PyTorch or matplotlib yet, run:

```bash
uv pip install torch matplotlib
```

### 1. Imports

```python
import torch
from torch import Tensor
import matplotlib.pyplot as plt
```

### 2. Create Synthetic Data

Let’s generate a simple regression task: predict $y = \sin(x)$ for $x \in [-2\pi, 2\pi]$.

```python
def generate_sine_data(n_samples: int = 100) -> tuple[Tensor, Tensor]:
    # Inputs: shape (N, 1)
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n_samples).unsqueeze(1)
    # Outputs: shape (N, 1)
    y = torch.sin(x)
    return x, y

X, y_true = generate_sine_data(300)
```

### 3. Initialize Parameters

```python
D_in = 1         # Input size
D_hidden = 10    # Hidden layer width
D_out = 1        # Output size

# Manual parameter initialization with requires_grad=True
W1 = torch.randn(D_in, D_hidden, requires_grad=True)
b1 = torch.zeros(D_hidden, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)
```

### 4. Training Loop

```python
learning_rate = 1e-2
n_epochs = 1000

losses: list[float] = []

for epoch in range(n_epochs):
    # Forward pass: linear, activation, linear
    h: Tensor = X @ W1 + b1    # shape: (N, D_hidden)
    h_relu: Tensor = torch.relu(h)
    y_pred: Tensor = h_relu @ W2 + b2  # shape: (N, D_out)

    # Compute and record loss
    loss: Tensor = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    # Backward pass
    loss.backward()

    # Update parameters using SGD (gradient descent)
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        # Zero gradients for next iteration
        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
    
    # Print progress every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}, Loss: {loss.item():.4f}")
```

### 5. Visualize the Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

# Model predictions
with torch.no_grad():
    h = X @ W1 + b1
    h_relu = torch.relu(h)
    y_pred = h_relu @ W2 + b2

plt.plot(X.numpy(), y_true.numpy(), label="True Function")
plt.plot(X.numpy(), y_pred.numpy(), label="Neural Net Prediction")
plt.legend()
plt.title("Neural Network Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot training loss
plt.figure()
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
```

---

## Exercises

Try the following exercises to deepen your understanding. Solutions are provided for reference!

---

### 1. Change the Activation Function

**Task:** Replace the ReLU activation with a Tanh activation. How does the model’s prediction change?

```python
# Replace torch.relu with torch.tanh everywhere in the training loop
learning_rate = 1e-2
n_epochs = 1000

# Re-initialize weights
W1 = torch.randn(D_in, D_hidden, requires_grad=True)
b1 = torch.zeros(D_hidden, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)

losses: list[float] = []

for epoch in range(n_epochs):
    h = X @ W1 + b1
    h_act = torch.tanh(h)                # <-- changed here
    y_pred = h_act @ W2 + b2

    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
```

---

### 2. Try a Wider Hidden Layer

**Task:** Increase $D_{hidden}$ from 10 to 50. What happens to the learned function?

```python
D_hidden = 50  # Increased hidden layer width

# Redefine and rerun the entire model
W1 = torch.randn(D_in, D_hidden, requires_grad=True)
b1 = torch.zeros(D_hidden, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)

losses: list[float] = []

for epoch in range(n_epochs):
    h = X @ W1 + b1
    h_relu = torch.relu(h)
    y_pred = h_relu @ W2 + b2

    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
```

---

### 3. Implement Early Stopping

**Task:** Add code to stop training when the loss is below 0.01.

```python
early_stop_threshold = 0.01
for epoch in range(n_epochs):
    h = X @ W1 + b1
    h_relu = torch.relu(h)
    y_pred = h_relu @ W2 + b2

    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    if loss.item() < early_stop_threshold:
        print(f"Early stop at epoch {epoch+1}, Loss: {loss.item():.4f}")
        break

    loss.backward()

    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
```

---

### 4. Add Noise to the Data

**Task:** Add Gaussian noise to $y$ and train the network again. How does the learned function change?

```python
X, y_true = generate_sine_data(300)
noise = 0.1 * torch.randn_like(y_true)
y_noisy = y_true + noise

# Use y_noisy instead of y_true in your training loop
```

---

## Summary

We've built a fully functioning, one-hidden-layer neural network *from scratch* using PyTorch tensors and manual parameter updates. Along the way, we've seen:

- How input data flows through the layers via linear transforms and activation functions
- How loss measures the "badness" of predictions
- How gradients allow us to know how to change each parameter to improve loss
- How manual gradient descent works with `.backward()` and `torch.no_grad()`
- How all these come together in a training loop!

---

## Key Takeaways

- **A neural network is just a sequence of linear transformations and nonlinear activations.**
- **You can build and train a neural net from scratch using only PyTorch tensors and basic math.**
- **The core ingredients: forward pass, loss calculation, backward pass (gradients), parameter update.**
- **Changing activation, hidden width, or the data itself can have a big impact on what the network learns.**
- **Understanding these basics is crucial before moving on to higher-level PyTorch abstractions like `nn.Module` and `torch.optim`.**

---

Next up, we’ll see *why* activation functions matter, and how they let neural networks discover richer patterns than a single linear function. See you there!

**USER:**


