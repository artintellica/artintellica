import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # support left add
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):  # left subtraction
        return Value(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data**2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    def __neg__(self):
        return self * -1

    def __pow__(self, power):
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        x = self.data
        out = Value(np.log(x), (self,), "log")

        def _backward():
            self.grad += (1 / x) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        out = Value(x if x > 0 else 0, (self,), "ReLU")

        def _backward():
            self.grad += (1 if x > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

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


def numerical_grad_f(x0, y0, eps=1e-6):
    """Finite-difference approximation for grad f(x, y) = x^2 + y^2 + 2xy."""
    def fx(x, y):
        return x**2 + y**2 + 2 * x * y

    dfdx = (fx(x0 + eps, y0) - fx(x0 - eps, y0)) / (2 * eps)
    dfdy = (fx(x0, y0 + eps) - fx(x0, y0 - eps)) / (2 * eps)
    return dfdx, dfdy


# Test values
x0, y0 = 1.2, -0.7

# Build the graph
x = Value(x0)
y = Value(y0)
f = x**2 + y**2 + 2 * x * y
f.backward()

# Print autodiff gradients
print("Autodiff gradients:")
print(f"∂f/∂x = {x.grad:.6f}")
print(f"∂f/∂y = {y.grad:.6f}")

# Numerical check
dfdx_num, dfdy_num = numerical_grad_f(x0, y0)
print("\nNumerical gradients (finite difference):")
print(f"∂f/∂x ≈ {dfdx_num:.6f}")
print(f"∂f/∂y ≈ {dfdy_num:.6f}")

# Analytic gradients for this function: ∂f/∂x = 2x + 2y, ∂f/∂y = 2y + 2x
print("\nAnalytic gradients:")
print(f"∂f/∂x = {2*x0 + 2*y0:.6f}")
print(f"∂f/∂y = {2*y0 + 2*x0:.6f}")
