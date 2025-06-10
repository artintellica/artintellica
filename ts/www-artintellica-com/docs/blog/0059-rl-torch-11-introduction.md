+++
title = "Learning Reinforcement Learning with PyTorch, Part 1.1: Introduction—The Course, the Blog, and Why PyTorch"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0059-rl-torch-11-introduction"
+++

Welcome! This is the first entry in **Artintellica's Reinforcement Learning
series**—a practical, code-driven open-source course where we build mathematical
foundations and RL agents **from vectors to deep Q-networks, all with PyTorch**.
Whether you've found us via GitHub, search, or your love of math and code,
_thank you_. This blog is intended for ambitious beginners who want more than a
"black box" approach.

What sets this course apart?

- **Code-first:** Every exercise uses Python, and everything is done in
  code—even math.
- **Open source:** Every post, example, and dataset is MIT licensed. Fork,
  modify, use!
- **Math-backed:** We'll use mathematics where appropriate—but always paired
  with concrete, hands-on coding.
- **PyTorch-powered:** PyTorch is now the dominant RL and deep learning research
  tool, with a gentle learning curve, intuitive tensor ops, and blazing
  performance on CPUs and GPUs.

**This first post** serves as both an orientation and your first hands-on steps:
installing PyTorch, running "Hello Tensors", and manipulating your first data
structures.

---

## What Will You Learn?

This post will help you:

- Understand how this RL course is structured (and what comes next)
- Install PyTorch on your system using the `uv` package manager
- Create and manipulate your first `torch.Tensor` objects
- Move tensors between CPU and GPU (if available)
- Run and interpret your first PyTorch computation

Each concept comes with code and hands-on exercises. By the end, you'll be ready
for the math and code engine that powers everything in modern deep learning
(and, soon, RL!).

---

## Why PyTorch? Why Not Just Numpy?

PyTorch is a numeric computation library like NumPy, but with several important
differences that make it essential for modern ML and RL:

1. **Automatic Differentiation:** Critical for optimization and training models.
2. **Hardware Acceleration:** Fast computation on CPUs, GPUs (CUDA), and Apple
   Silicon (MPS) with almost no code changes.
3. **Dynamic Graphs:** Models can be defined and modified "on the fly", perfect
   for research and RL.
4. **Neural Network Utilities:** Everything from layers to RL-specific
   frameworks.
5. **Community & Ecosystem:** Used everywhere—if you can do it in deep RL, you
   can probably do it in PyTorch.

Throughout this course, you'll see how PyTorch lets us build RL agents—from
table-based to neural-based—with minimal friction.

---

## Mathematics: How PyTorch Relates to Vectors and Matrices

At its core, PyTorch's **Tensor** object generalizes the matrix and vector
concepts from linear algebra. In this blog, everything you learned about scalars
($a$), vectors ($\mathbf{x}$), and matrices ($A$) will correspond directly to
PyTorch `Tensor` objects—uniquely enabling you to:

- Write code that _is_ the math
- Run experiments interactively
- Train and debug agents at every complexity level

We'll often use math notation ($\mathbf{x}$, $A$, etc.) side-by-side with its
exact code representation (`torch.Tensor`). As you encounter concepts like dot
product, matrix multiplication, or vector norms, we'll show both the formula and
its PyTorch incarnation.

---

## Installing PyTorch (with `uv`)

> ℹ️ For this blog, we'll assume you use Python 3.9 or later on Linux, macOS, or
> Windows. Use VS Code, Jupyter, or any Python IDE you like.

We recommend [uv](https://github.com/astral-sh/uv) for fast, reproducible,
modern package management.

### **Step 1:** Install `uv` (if not already installed)

```sh
pip install --upgrade uv
```

### **Step 2:** Create a new project directory and initialize

```sh
mkdir rl-with-pytorch && cd rl-with-pytorch
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### **Step 3:** Install PyTorch + matplotlib (for plots)

Find your install command at:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

For most CPU-only systems:

```sh
uv pip install torch torchvision torchaudio matplotlib
```

For CUDA-enabled GPUs, use the correct `torch` version:

```sh
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install matplotlib
```

### **Verify Installation**

Open Python:

```python
import torch
print("Torch version:", torch.__version__)
print(torch.rand(2, 2))
```

If this prints the version and a $2\times 2$ tensor of random numbers, you're
ready!

---

## Python Demonstrations

Let's walk through key tensor operations.

### **Demo 1: Create a Tensor and Print Its Data, Shape, dtype**

```python
import torch

# Create a 1D tensor of floats, from 0 to 4
x: torch.Tensor = torch.arange(5, dtype=torch.float32)
print("x:", x)  # Tensor data
print("Shape:", x.shape)
print("Dtype:", x.dtype)
```

### **Demo 2: Move Tensor to GPU (if available) or MPS (Apple Silicon)**

```python
# Detect GPUs or MPS device (Apple Silicon)
if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

x_gpu: torch.Tensor = x.to(device)
print("x is on device:", x_gpu.device)
```

### **Demo 3: "Hello World"—Elementwise Addition**

Elementwise addition is just like with math or NumPy: $z_i = x_i + y_i$

```python
y: torch.Tensor = torch.ones(5, dtype=torch.float32, device=device)
z: torch.Tensor = x_gpu + y  # elementwise add
print("z:", z)
```

---

## Exercises

Try these yourself! (Start from a new Python file, or use a Jupyter notebook.)

### **Exercise 1:** Install PyTorch and Verify

- Install using `uv` as above. Write Python code to import `torch` and print its
  version.

### **Exercise 2:** Create Your First Tensor

- Create a tensor of shape `(4, 3)` (4 rows, 3 cols) filled with random numbers.
- Print the tensor, its `shape`, and `dtype`.

### **Exercise 3:** Move Tensor Between Devices

- Detect if a GPU or MPS device is available.
- Move a tensor to CPU, then to device, and print its `.device` at each stage.

### **Exercise 4:** Hello World Addition

- Create two tensors of size `(6,)` (one with all zeros, one with all ones).
- Add them and print the result.

---

**Sample Starter Code for Exercises:**

```python
import torch

# EXERCISE 1
print("PyTorch version:", torch.__version__)

# EXERCISE 2
a: torch.Tensor = torch.randn(4, 3)
print("Tensor a:", a)
print("Shape:", a.shape)
print("Dtype:", a.dtype)

# EXERCISE 3
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
b: torch.Tensor = torch.arange(6)
print("Before:", b.device)
b = b.to(device)
print("After:", b.device)

# EXERCISE 4
zero: torch.Tensor = torch.zeros(6)
one: torch.Tensor = torch.ones(6)
sum_: torch.Tensor = zero + one
print("Sum:", sum_)
```

---

## Conclusion

In this post, you've:

- Oriented yourself to the **Artintellica RL with PyTorch course**
- Learned how to quickly set up a PyTorch environment with `uv`
- Explored PyTorch's powerful tensor abstraction as a bridge between math and
  code
- Written your first PyTorch code for tensor creation, device moves, and
  addition
- Practiced with beginner exercises you'll build on throughout the course

**Next:** We'll dig into the fundamentals of vectors and scalars in PyTorch,
learning to manipulate, index, and visualize them—the real foundation for RL
algorithms.
