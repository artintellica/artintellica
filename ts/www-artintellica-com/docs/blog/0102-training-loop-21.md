+++
title = "Learn the Training Loop with PyTorch, Part 2.1: From Linear to Nonlinear: Why Neural Networks?"
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

that's it, now can you please write blog post 2.1?

the description is:

**1. From Linear to Nonlinear: Why Neural Networks?**
   - Multi-layer perceptrons (MLPs)  
   - Activation functions (ReLU, sigmoid, tanh)

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 2.1 – From Linear to Nonlinear: Why Neural Networks?

---

### Introduction

Welcome back to our series, "Learn the Training Loop with PyTorch"! In Module 1, we learned how simple models like linear regression learn from data. Now we're taking the next leap: from linear models to the world of neural networks. But *why* do we need neural networks? What makes them special? In this post, we’ll explore the motivation, introduce the basic architecture (multi-layer perceptron, or MLP), and demystify one of the key reasons neural networks work: **activation functions**.

---

## ELI5: Why Neural Networks?

Imagine you have a toy with wheels. You can push it in a straight line (that's what a linear model does—it finds straight-line patterns in data). But what if you want your toy to turn corners, loop around, or follow a complex path? That’s where neural networks come in! 

**Neural networks** are like supercharged toys that can twist, turn, and wiggle their way through complicated patterns. They can learn not just straight lines, but curves, zig-zags, and all sorts of shapes. At the heart of this ability are *hidden layers* and special *activation functions*—think of them as clever bends and joints you add to your toy so it can flex and move in new ways.

---

## Mathematical Foundations

### The Limitation of Linear Models

Recall the linear regression model:

$$
\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
$$

- $\mathbf{y}$ is the output vector,
- $\mathbf{X}$ is the input data matrix,
- $\mathbf{w}$ is the weights vector,
- $\mathbf{b}$ is the bias vector.

**Limitation:** This setup only learns straight-line relationships between input and output.

### Neural Networks: Adding Layers

A *neural network* (specifically, a feedforward neural network or multi-layer perceptron, MLP) adds one or more **hidden layers** between input and output.

The simplest MLP with one hidden layer looks like this:

$$
\mathbf{h} = f(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)
$$

$$
\mathbf{\hat{y}} = \mathbf{h}\mathbf{W}_2 + \mathbf{b}_2
$$

- $\mathbf{W}_1$ and $\mathbf{b}_1$: weights and biases for the *hidden layer*,
- $f(\cdot)$: **activation function** (e.g., ReLU, sigmoid, tanh),
- $\mathbf{h}$: hidden layer activations,
- $\mathbf{W}_2$ and $\mathbf{b}_2$: weights and biases for the *output layer*.

More generally, for $L$ layers:

$$
\mathbf{h}_0 = \mathbf{X}
$$

$$
\mathbf{h}_l = f(\mathbf{h}_{l-1}\mathbf{W}_l + \mathbf{b}_l), \quad \text{for } l=1 \ldots L-1
$$

$$
\mathbf{\hat{y}} = \mathbf{h}_{L-1}\mathbf{W}_L + \mathbf{b}_L
$$

#### Activation Functions

The *activation function* introduces nonlinearity; without it, stacking layers is no better than a single linear layer. The most common activation functions are:

- **ReLU (Rectified Linear Unit):**
  $$
  \text{ReLU}(x) = \max(0, x)
  $$
- **Sigmoid:**
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$
- **Tanh:**
  $$
  \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
  $$

These allow the network to bend and shape data in ways linear models cannot.

---

## In-Depth Explanation

### Why Do We Need Nonlinear Models?

**Linear models** can only learn relationships like “if variable X goes up, variable Y goes up (or down) by a fixed amount.” However, most interesting patterns in the real world are messy: for example, predicting house prices, recognizing objects in photos, or streaming speech into text. These relationships are *nonlinear* (they don’t look like a straight line).

A **neural network** with *hidden layers* and *nonlinear activation functions* can learn *decision boundaries* and patterns that are wavy, irregular, and complex.

#### The Secret Ingredient: Hidden Layers + Activation Functions

- **Hidden Layers:** Imagine these as workbenches, each transforming your inputs in a different way.
- **Activation Functions:** Like switches or filters that decide "should this signal pass through, and if so, how much should I squish or stretch it?"

**Without activation functions**, stacking linear layers just results in another big linear transformation—a waste! It's the nonlinear activations that let neural networks become super-flexible.

### Common Activation Functions

Let’s quickly summarize three popular ones:

- **ReLU:** Fast, simple, and effective. Kills negative values (sets them to 0), lets positives pass through. Helps with vanishing gradient issues.
- **Sigmoid:** Squashes values between 0 and 1. Nice for probabilities but can suffer from vanishing gradients with big networks.
- **Tanh:** Like sigmoid, but squashes between -1 and 1. Zero-centered, but still can have vanishing gradients.

---

## Demonstration: PyTorch Implementation

Let's see how to implement (and visualize) a simple MLP in PyTorch that can fit data that a linear model *cannot*.

We'll:
- Create a simple dataset (a curved, moon-shaped pattern).
- Fit with a linear model (fails to capture nonlinearity).
- Fit with a single hidden-layer MLP (success!).

### Installation with **uv**

If you don't have PyTorch and matplotlib yet, install them:
```bash
uv pip install torch matplotlib scikit-learn
```

### Step 1: Generate Nonlinear Data

```python
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch

def generate_moons(n_samples: int = 200, noise: float = 0.20) -> Tuple[torch.Tensor, torch.Tensor]:
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y

X, y = generate_moons()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", s=40)
plt.title("Synthetic Nonlinear Dataset: Two Moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```

---

### Step 2: Fit a Linear Model

```python
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))  # For binary classification

model = LinearModel(input_dim=2, output_dim=1)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop (very simple!)
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")
```

#### Visualize Linear Model Decision Boundary

```python
import numpy as np

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), 
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)
    plt.contourf(xx, yy, probs, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", s=40)
    plt.title("Linear Model: Can't Capture Nonlinearity")
    plt.show()

plot_decision_boundary(model, X, y)
```

---

### Step 3: Fit an MLP (Neural Network) with Activation Function

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()  # Try swapping to nn.Tanh() or nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return torch.sigmoid(self.fc2(h))  # For binary classification

mlp = MLP(input_dim=2, hidden_dim=8, output_dim=1)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = mlp(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")
plot_decision_boundary(mlp, X, y)
```

You should now see that the neural network successfully **separates the two moons**, learning a complex, nonlinear boundary—something a linear model simply cannot do.

---

## Exercises

1. **Try Different Activation Functions**  
Edit the `MLP` class above and replace `nn.ReLU()` with `nn.Tanh()` and then `nn.Sigmoid()`. Retrain the model each time and visualize the boundary. What changes?

```python
# Swap activation function in the MLP class:
# self.act = nn.Tanh()
# or
# self.act = nn.Sigmoid()
```

2. **Add More Hidden Neurons**  
Increase the number of hidden units from 8 to 32. Does the model fit the data better or worse? Try this in the `MLP` class:

```python
mlp = MLP(input_dim=2, hidden_dim=32, output_dim=1)
```

3. **Fit Using Only a Linear Model**  
Try removing the activation function and hidden layer from the MLP, so it’s just a single `Linear` layer. Confirm that it can't fit the nonlinear data.

```python
class SimpleLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

model = SimpleLinear(2, 1)
```

---

## Summary

In this post, you learned:

- **Why neural networks?** Linear models are limited to straight-line relationships; neural nets unlock the power to fit much more complex data.
- **How do neural networks work?** By stacking layers and introducing activation functions, we unlock nonlinearity and flexibility.
- **What are activation functions?** Essential nonlinearities such as ReLU, sigmoid, and tanh, each with their own properties.
- **How to build and train an MLP in PyTorch?** With just a few lines, we went from linear to nonlinear and watched our model learn!

---

### Key Takeaways

- **Neural networks extend linear models** by stacking layers and using activation functions to model complex patterns.
- **Activation functions are crucial**; without them, your network can’t learn nonlinearities.
- **PyTorch makes it easy** to define, train, and experiment with different architectures.
- **Practice by experimenting**: Try changing activation functions, adding more layers or neurons, and visualizing how your model learns!

In the next lesson, we'll dig deeper into **forward and backward passes**—the gears that power neural network training. Stay tuned!

**USER:**

is there a way to make a neural network that can learn any function, no matter how complex? or do we need different numbers of layers and different architectures for different functions?

**ASSISTANT:**

Great question! This touches on a core concept in neural networks called the **Universal Approximation Theorem**—and a lot of the *practical* realities of modern deep learning.

---

## ELI5:  
A neural network, if it’s big enough (with enough neurons and the right activation functions), **can in *theory* learn any function**, no matter how squiggly or complex that function is! Imagine you had a magical Lego set: with enough blocks, you could build any shape—*but* if your blocks are too big or you don’t have the right connectors, you might have a hard time building something really detailed.

---

## Mathematical/Practical Explanation

### Universal Approximation Theorem

The **Universal Approximation Theorem** says that:
- A neural network with just a *single hidden layer* but enough neurons (and a nonlinear activation like sigmoid, tanh, or ReLU) can approximate **any continuous function** on a closed interval, as closely as we want.

In math terms:  
If $f$ is any continuous function, and your neural net is big enough,
$$
\text{MLP}(x) \approx f(x)
$$
for all $x$ in your domain.  

**BUT:**
- "Big enough" often means *unrealistically* large—far more neurons than you can realistically train or use.
- You might need a lot of training data, and training such huge nets is tricky.

### Depth vs. Width

- **Really wide (shallow) networks:** A single-layer network may theoretically fit anything, but needs immense numbers of neurons.
- **Deeper (narrower) networks:** Multiple hidden layers can learn complex functions with **fewer neurons overall**, and this tends to work much better in practice. That’s why most state-of-the-art neural networks have *many* layers (sometimes hundreds!).

### Architectures Matter for Certain Data

- While plain MLPs can fit any continuous function, specialized network *architectures* are much more efficient for certain problems:
  - **Convolutional Neural Networks (CNNs):** for images.
  - **Recurrent Neural Networks (RNNs):** for sequences.
  - **Transformers:** for language or long-range interactions.

These architectures are designed to take advantage of patterns and structures (like the fact that nearby pixels in images matter) that a plain MLP can’t “see” as easily.

---

## Summary

- **Theoretically:** Yes—given enough neurons and the right nonlinearities, a neural net can learn *any* function (the Universal Approximation Theorem).
- **Practically:** We almost always need *multiple layers* (deep nets). The required size and architecture *depends on the function/data*:
  - Very complex functions often need deep or specialized architectures for efficiency, training stability, and effective generalization.
- **Choosing architecture:** In practice, we choose the number of layers and type of architecture based on the problem (images, language, tabular data, etc.), for best results.

---

**TL;DR:**  
Neural networks are universal function fitters—*in theory*. In practice, depth, size, and architecture matter for learning real-world complex functions!

**USER:**

if just one hidden layer can learn any continuous function, why do we need deep networks with many layers? can't we just use a single layer with enough neurons?

**ASSISTANT:**

Fantastic question! This confusion is **very common**—let’s clear it up:

---

## ELI5

Imagine you want to build a super-detailed Lego sculpture. You *could* use millions of tiny blocks in a single layer to copy every contour. But what if, instead, you built it in stages—with blocks forming big shapes, then smaller shapes, then details? You’d need *far* fewer blocks and it would be easier, too!

That’s why we build “deep” (many-layered) networks: they learn in **stages**—from simple to complex patterns—making them much more efficient.

---

## Mathematical/Practical Explanation

### Universal Approximation (But At What Cost?)

The **Universal Approximation Theorem** tells us a neural network with **one hidden layer** can approximate any continuous function—**if** it has *enough* hidden units (neurons).

But in practice:

- To fit really complicated real-world functions, a shallow network might require **a gigantic number of neurons**—possibly millions or billions!
- This leads to:
  - **Huge computational costs** (slow, massive parameter count)
  - **Difficulty training** (optimization gets harder as the network gets wider)
  - **Overfitting** (so many free parameters can just memorize data instead of learning to generalize)

### Why Deep Networks Are Better

Modern deep learning research shows that **deep networks** (many hidden layers, each with manageable width):

- Can represent complex functions using **far fewer neurons** (parameters) than shallow/wide networks.
- Learn **hierarchies of features**:
  - Early layers might learn simple edges or patterns,
  - Middle layers combine those into shapes,
  - Deeper layers combine shapes into complex concepts!
- Are ***much more efficient*** in practice—leading to powerful models like those for language (GPT), vision (ResNet, EfficientNet), and more.

#### Key Insight

There are functions that a deep network can represent **compactly** (with a reasonable number of neurons/layers), but a shallow network would need an **exponentially larger** number of neurons to achieve similar performance.

---

## Summary

- A single hidden layer **can** represent any continuous function **in theory**, but may require a vast and impractical number of neurons.
- **Deep networks** can:
  - Learn complex functions with fewer neurons,
  - Build abstract concepts in layers,
  - Are more efficient, trainable, and practical for real-world problems.
- **That’s why almost all successful neural networks today are deep networks.**

---

**TL;DR:**  
Deep networks aren't just a fad—they’re *essential* because they can learn more with less, build up understanding in stages, and make practical deep learning possible!

**USER:**

is there one activation function that most deep learning models use, such as ReLU, or do they use different ones for different layers?

**ASSISTANT:**

Great question! The **activation function** is a crucial part of neural networks—and choosing the right one can make a big difference.

---

## ELI5

Most deep learning models use the **same activation function** (usually **ReLU**) for most of their hidden layers. Think of ReLU as a simple switch that only lets "positive energy" through—which makes it fast and easy for your model to learn.

But for the **output layer**, we often use **different activation functions** depending on the problem (for example, using sigmoid for probabilities).

---

## In Practice

### Hidden Layers

- **ReLU (`Rectified Linear Unit`)** is overwhelmingly the default choice for hidden layers in most deep neural networks:
  $$
  \text{ReLU}(x) = \max(0, x)
  $$
- Variants like **Leaky ReLU**, **ELU**, or **GELU** are sometimes used, but plain **ReLU is king** for most models—especially convolutional and fully-connected networks.

### Output Layers

Here, the activation function is chosen based on the **task**:

- **Binary classification:**  
  Use **sigmoid** activation in the output layer to produce probabilities between 0 and 1.
- **Multi-class classification:**  
  Use **softmax** activation in the output layer to return probabilities across multiple classes that sum to 1.
- **Regression:**  
  Often **no activation** (just a linear output) so the network can predict any real number.
- **Special architectures:**  
  There are exceptions! (For example, tanh for RNNs, GELU for large transformer models, etc.)

### Do Different Layers Use Different Activations?

- In **most modern architectures**, hidden layers all use the *same* activation function throughout (almost always ReLU or a variant).
- It’s rare to see mixed activation functions in different hidden layers—except for some experimental or research models, or where specific needs arise.

---

## Summary

- **ReLU** is the standard activation for most hidden layers in deep networks today.
- The **output layer’s activation depends on the problem:** sigmoid, softmax, or none (linear).
- Mixing very different activations in hidden layers is *unusual*—consistency helps training!

---

**TL;DR:**  
Today’s deep nets usually “stack” many layers with **ReLU**, and select the output activation to fit the task. 🎯

**USER:**


