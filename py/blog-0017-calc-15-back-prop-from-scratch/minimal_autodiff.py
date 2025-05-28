# calc-15-backprop-from-scratch/minimal_autodiff.py
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        out = Value(
            self.data + (other.data if isinstance(other, Value) else other),
            (self, other),
            "+",
        )

        def _backward():
            self.grad += out.grad
            if isinstance(other, Value):
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(
            self.data * (other.data if isinstance(other, Value) else other),
            (self, other),
            "*",
        )

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
        out = Value(self.data**power, (self,), f"**{power}")

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


a = Value(2.0)
b = Value(-3.0)
c = a * b + a**2
d = c.tanh()
d.backward()
print(f"d = {d.data:.5f}, ∂d/∂a = {a.grad:.5f}, ∂d/∂b = {b.grad:.5f}")
