+++
title = "Learning Reinforcement Learning with PyTorch, Part 1.1: The Course, the Blog, and Why PyTorch"
author = "Artintellica"
date = "2024-06-09"
+++

Welcome to Artintellica's open-source series on learning reinforcement learning with PyTorch! If you’ve followed along so far, you’ve primed yourself with linear algebra, calculus, and deep learning—so you’re in the perfect position to step into the world of reinforcement learning (RL). In this post, we’ll set the stage for the journey ahead: what this blog series is about, why PyTorch is our framework of choice, and how you’ll rapidly build the mathematical and coding fluency needed for deep RL research and projects.

We'll wrap up with hands-on Python exercises to get your system configured and your PyTorch tensors up and running.

---

## Why Learn Reinforcement Learning?

Reinforcement learning lies at the intersection of machine learning, control theory, and behavioral psychology. It powers game-playing AIs, robotics, recommendation engines, and much more. Unlike supervised learning—where the “right answer” is given—RL agents must **learn by interacting**: taking actions, observing results, and adapting to maximize long-term rewards.

A typical RL workflow looks like this:

- Agent observes the current state,
- Chooses an action,
- Receives a reward and observes the next state,
- Learns from the experience.

Soon, you’ll formalize these concepts mathematically, but let’s first set up the tools.

---

## Why PyTorch?

### 1. Research Friendly, Pythonic, and Highly Flexible

PyTorch is the dominant library for deep RL research, thanks to its:

- **Intuitive API:** Similar to NumPy, but with powerful GPU acceleration.
- **Dynamic Computation Graphs:** You can write regular Python flow control (loops, conditionals) and the computation graph is built on-the-fly.
- **Integration with RL Frameworks:** Leading RL packages (e.g., Stable-Baselines3, RLlib) embrace PyTorch.
- **Autograd:** Native, painless automatic differentiation for backprop and RL gradient methods.

### 2. Seamless CPU/GPU/MPS Support

With minimal code changes, your models can run on laptops, desktops, or cloud GPUs/Apple Silicon—crucial for RL, where speedups are substantial.

### 3. Ecosystem and Community

Vast PyTorch resources, tutorials, and extensions mean help is always close by, and you’ll find RL codebases and papers almost universally include PyTorch implementations.

---

## Mathematics: Tensors, the Building Blocks

Most RL and deep learning code operates on **tensors**. A tensor generalizes the concept of scalars, vectors, and matrices:

- **Scalar:** A single number ($x \in \mathbb{R}$), 0D.
- **Vector:** An array of numbers ($\mathbf{x} \in \mathbb{R}^n$), 1D.
- **Matrix:** 2D array.
- **Tensor:** Generalization to $n$ dimensions (ND-array).

PyTorch’s `torch.Tensor` type will be your canvas for every operation, from basic arithmetic to backpropagation. Throughout the course, you’ll manipulate tensors for everything: states, actions, gradients, policies, and value functions.

---

## PyTorch Installation and Hello Tensor: Step-by-Step

Learning ML by *doing* is the best way—and PyTorch makes it fun. Let’s get your environment set up and quickly see tensors in action.

### Python Demonstrations

#### **1. Installing PyTorch**

Go to the [official install page](https://pytorch.org/get-started/locally/) and select your options (OS, package manager, CUDA version). For typical CPU-only install via pip:

```bash
pip install torch torchvision torchaudio
```

If you’re using a GPU (NVIDIA), be sure to install the right CUDA version!

#### **2. Creating & Inspecting a Tensor**

```python
import torch

# Create a 1D tensor (vector) of floats
x = torch.tensor([1.0, 2.0, 3.0])

print("x:", x)
print("Shape of x:", x.shape)
print("Data type of x:", x.dtype)
```

#### **3. Moving Tensors Between Devices**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

x_device = x.to(device)
print("Tensor device:", x_device.device)
```

PyTorch will automatically move the tensor to GPU if available. On Apple Silicon, it may use MPS (Metal Performance Shaders), so check for `"mps"` as well.

#### **4. Hello World: Elementwise Tensor Addition**

```python
# Elementwise addition
y = torch.tensor([4.0, 5.0, 6.0])

z = x + y  # Or torch.add(x, y)
print("x + y =", z)
```

---

## Python Exercises

Ready to get your hands dirty? Try these foundational exercises. Paste the code into a notebook or script, experiment, and tweak!

---

**Exercise 1:** *Install PyTorch and verify the installation.*

- Install using pip or conda.
- Import torch and print its version.

```python
import torch
print("PyTorch version:", torch.__version__)
```

---

**Exercise 2:** *Create your first tensor and print its shape and dtype.*

- Make a tensor with values `[10, 20, 30, 40]`.
- Print its shape and data type.

---

**Exercise 3:** *Move a tensor between CPU and (if available) GPU/MPS, and print device info.*

- Detect available devices and move your tensor.
- Print the tensor's current device.

---

**Exercise 4:** *Run and explain a “Hello World” example—elementwise addition in PyTorch.*

- Create two tensors and add them.
- Print the result.
- Explain the result in a comment or markdown cell.

---

## Conclusion

You’re now equipped to use PyTorch and manipulate tensors—the foundation of everything you'll do in RL (and deep learning in general). In the next post, we’ll take a step deeper into vectors and scalars—building up our mathematical and computational toolkit, always with hands-on code.

If you have trouble with installations (CPU, GPU, or Apple MPS), check the official docs or share your error messages in the comments—open-source is all about helping each other! 

See you in the next lesson, where we’ll explore vectors, scalars, and more with code and visualizations.

**Happy coding, and welcome to the adventure of reinforcement learning with PyTorch!**
