+++
title = "Calculus 15: Backpropagation from Scratch — How Reverse-Mode Autodiff Works"
date = "2025‑05‑28"
author = "Artintellica"
+++

> _“Backpropagation is just the chain rule, efficiently applied in reverse.”_

---

## 1 · What is Reverse-Mode Autodiff?

- **Forward mode:** Tracks derivatives as you compute outputs.
- **Reverse mode:** Records the computation graph, then computes gradients
  backward from output to inputs.

Reverse-mode (backprop) is ideal when you have **one output, many inputs**
— like loss gradients in deep learning.

---

## 2 · Why ML Engineers Care

- **Backprop** is the engine behind all modern deep learning.
- PyTorch, TensorFlow, JAX, etc. all implement reverse-mode AD.
- Understanding it lets you debug, optimize, and innovate.

---

## 3 · A Minimal Autodiff Engine

We’ll define a `Value` class to represent numbers in the computation graph,
storing both value and gradient. Each operation links nodes together.

### **Forward pass:** Evaluate output.

### **Backward pass:** Use the chain rule to propagate gradients.

```python
# calc-15-backprop-from-scratch/minimal_autodiff.py
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        out = Value(self.data + (other.data if isinstance(other, Value) else other), (self, other), "+")
        def _backward():
            self.grad += out.grad
            if isinstance(other, Value):
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * (other.data if isinstance(other, Value) else other), (self, other), "*")
        def _backward():
            self.grad += (other.data if isinstance(other, Value) else other) * out.grad
            if isinstance(other, Value):
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")
        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    if isinstance(child, Value):
                        build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

**Usage Example:**

```python
import numpy as np
a = Value(2.0)
b = Value(-3.0)
c = a * b + a ** 2
d = c.tanh()
d.backward()
print(f"d = {d.data:.5f}, ∂d/∂a = {a.grad:.5f}, ∂d/∂b = {b.grad:.5f}")
```

---

## 4 · Check: Compare to PyTorch

```python
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = a * b + a ** 2
d = torch.tanh(c)
d.backward()
print(f"d = {d.item():.5f}, ∂d/∂a = {a.grad.item():.5f}, ∂d/∂b = {b.grad.item():.5f}")
```

---

## 5 · What Did We Build?

- Our engine constructs a computation graph, tracks values, and propagates
  gradients using the chain rule.
- With a few more ops (sub, div, exp, log, relu, etc.), you have the backbone of
  all deep learning frameworks.

---

## 6 · Exercises

1. **Add More Ops:** Implement subtraction, division, exp, log, relu, etc. in
   the `Value` class.
2. **Custom Function:** Build a function $f(x, y) = x^2 + y^2 + 2xy$, backprop,
   and check the gradients numerically.
3. **Vectorization:** Extend the engine to support vector inputs (lists of
   `Value`s) and backprop through a small neural network.
4. **Visualization:** Write a function to print or plot the computation graph
   for a `Value` node.
5. **Reverse-Mode vs. Forward-Mode:** For $f(x, y, z) = x + y + z$, compare
   reverse-mode AD (backprop) and forward-mode (Jacobian).

Put solutions in `calc-15-backprop-from-scratch/` and tag `v0.1`.

---

**Next:** _Calculus 16 — Automatic Differentiation, Modern ML, and You._
