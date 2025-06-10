+++
title = "Learn Reinforcement Learning with PyTorch, Part 2.2: Autograd—Automatic Differentiation Demystified"
author = "Artintellica"
date = "2024-06-10"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0069-rl-torch-22-autograd"
+++

## Introduction

Welcome back to Module 2 of Artintellica’s RL with PyTorch! Previously, you
implemented gradient descent "by hand." Now it’s time to take the next leap:
**automatic differentiation**. PyTorch’s `autograd` is its most magical,
productivity-enhancing feature. It allows you to compute gradients of complex,
multi-step functions automatically—fueling modern deep learning, RL policies,
and more.

In this post you will:

- Learn how PyTorch keeps track of computations for backpropagation.
- Use `requires_grad=True` to enable automatic gradient tracking.
- Compute gradients for both simple and multi-variable functions.
- See why (and how) to "zero" gradients in optimization loops.
- Manually verify autograd’s gradient calculations for trust and understanding.

Let’s open the black box, see how autograd works, and demystify gradients for
good.

---

## Mathematics: What is Automatic Differentiation?

**Automatic differentiation** is a technique where the library automatically
records all operations performed on tensors with `requires_grad=True` to build a
computation graph. When you call `.backward()`, PyTorch traces this graph
**backwards** from your output (often the loss) and computes derivatives with
respect to all required inputs using the **chain rule**.

For $f(x, y) = x^2 + y^3$:

- $\frac{\partial f}{\partial x} = 2x$
- $\frac{\partial f}{\partial y} = 3y^2$

Autograd does this for you, no matter how complex your function.

---

## Python Demonstrations

### Demo 1: Mark a Tensor as `requires_grad=True` and Compute Gradients

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
f = x**2
f.backward()  # Compute the gradient ∂f/∂x at x=3

print("Value of f(x):", f.item())
print(
    "Gradient at x=3 (df/dx):",
    x.grad.item() if x.grad is not None else "No gradient computed",
)
```

---

### Demo 2: Multi-variable Function—$f(x, y) = x^2 + y^3$

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(-1.0, requires_grad=True)
f = x**2 + y**3
f.backward()  # Compute gradients for both x and y

print("f(x, y):", f.item())
print("df/dx at (x=2):", x.grad.item())  # Should be 4.0
print("df/dy at (y=-1):", y.grad.item())  # Should be 3*(-1)**2 = 3.0
```

---

### Demo 3: Zero Gradients—Why and How

PyTorch accumulates gradients by default. If you don’t zero them in each
optimization step, you’ll add up gradients across steps, which is almost always
undesirable.

```python
w = torch.tensor(1.0, requires_grad=True)

for step in range(2):
    f = (w - 2)**2  # Simple loss
    f.backward()
    print(f"Step {step}: w.grad = {w.grad.item()}")
    w.grad.zero_()  # Zero the gradient for the next step
```

**Why zero?**

- Each call to `.backward()` adds to `.grad`—unless you zero, your gradients
  accumulate and your parameter updates become wrong!

---

### Demo 4: Manually Verify PyTorch’s Gradient for a Small Example

Let’s explicitly check the numbers.

```python
# Let’s use f(x) = x^2 at x=4.0, df/dx should be 8.0
x = torch.tensor(4.0, requires_grad=True)
f = x**2
f.backward()
print("PyTorch grad:", x.grad.item())  # Should be 8.0

# Manual calculation
manual_grad = 2 * x.item()  # 2 * 4.0 = 8.0
print("Manual grad:", manual_grad)
```

---

## Exercises

Apply and verify your knowledge:

### **Exercise 1:** Mark a Tensor as `requires_grad=True` and Compute Gradients

- Create a tensor $z = 5.0$ with `requires_grad=True`.
- Let $f(z) = 3z^2 + 4z$.
- Compute $f(z)$ and call `.backward()`.
- Print the gradient `z.grad`.

---

### **Exercise 2:** Calculate Gradients for a Multi-variable Function

- Let $x=1.5$, $y=-2.0$, both with `requires_grad=True`.
- Define $f(x,y) = 5x^2 + xy + 2y^3$.
- Compute $f(x, y)$ and call `.backward()`.
- Print `x.grad` and `y.grad`.

---

### **Exercise 3:** Zero Gradients and Explain Why This is Necessary in Training Loops

- Re-run the previous gradient calculation **twice in a row** without zeroing
  gradients.
- Observe the value of `.grad` on the second `.backward()`.
- Now use `.grad.zero_()` after each `.backward()` and verify `.grad` is correct
  each time.

---

### **Exercise 4:** Manually Verify PyTorch’s Computed Gradients for a Small Example

- Use $f(x) = 7x^2$ at $x = 3.0$.
- Compute the gradient using autograd and by hand.
- Ensure the answers match.

---

### **Sample Starter Code for Exercises**

```python
import torch

# EXERCISE 1
z = torch.tensor(5.0, requires_grad=True)
f = 3 * z**2 + 4 * z
f.backward()
print("Gradient df/dz:", z.grad.item())  # Should be 6*z + 4 = 34

# EXERCISE 2
x = torch.tensor(1.5, requires_grad=True)
y = torch.tensor(-2.0, requires_grad=True)
f = 5 * x**2 + x*y + 2*y**3
f.backward()
print("df/dx:", x.grad.item())  # Should be 10*x + y = 10*1.5 + (-2) = 13
print("df/dy:", y.grad.item())  # Should be x + 6*y^2 = 1.5 + 6*4 = 25.5

# EXERCISE 3
x.grad.zero_(); y.grad.zero_()  # Try commenting this out and see accumulation!
f = 5 * x**2 + x*y + 2*y**3
f.backward()
print("After zeroing gradients then backward:")
print("df/dx:", x.grad.item())
print("df/dy:", y.grad.item())

# EXERCISE 4
x2 = torch.tensor(3.0, requires_grad=True)
f2 = 7 * x2**2
f2.backward()
print("PyTorch grad:", x2.grad.item())  # Should be 14*x2 = 42
print("Manual grad:", 14 * x2.item())
```

---

## Conclusion

PyTorch's **autograd** is the engine behind every neural network, policy
gradient, or optimizer. You’ve learned how to:

- Compute gradients automatically for any scalar function.
- Handle multi-variable functions and check their gradients.
- Properly zero gradients in training loops to prevent subtle bugs.
- Manually verify autograd results and ensure your math lines up!

**Next:** You’ll see how all this feeds into **loss functions and surfaces**:
how we measure error and steer optimization in deep learning and RL.

_Experiment with more complicated functions and see how PyTorch does the
calculus for you! See you in Part 2.3!_
