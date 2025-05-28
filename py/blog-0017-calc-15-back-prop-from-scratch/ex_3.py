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


def dot(xs, ws, b):
    """Compute dot product plus bias: xs, ws are lists of Value, b is Value."""
    out = sum([x * w for x, w in zip(xs, ws)], b)
    return out


def tiny_mlp(xs, params):
    """A 2-2-1 network with tanh activations."""
    # Unpack weights/biases (all Value)
    # Layer 1: 2 neurons, each takes 2 inputs
    w11, w12, b1 = params["w11"], params["w12"], params["b1"]
    w21, w22, b2 = params["w21"], params["w22"], params["b2"]
    # Output layer: 1 neuron, 2 inputs
    v1, v2, c = params["v1"], params["v2"], params["c"]

    # Hidden layer
    h1 = (xs[0] * w11 + xs[1] * w12 + b1).tanh()
    h2 = (xs[0] * w21 + xs[1] * w22 + b2).tanh()
    # Output (linear)
    out = h1 * v1 + h2 * v2 + c
    return out


# Example input
x0, x1 = 1.5, -0.8
inputs = [Value(x0), Value(x1)]

# Example random parameters (Value objects)
np.random.seed(42)
params = {
    "w11": Value(np.random.randn()),
    "w12": Value(np.random.randn()),
    "b1": Value(np.random.randn()),
    "w21": Value(np.random.randn()),
    "w22": Value(np.random.randn()),
    "b2": Value(np.random.randn()),
    "v1": Value(np.random.randn()),
    "v2": Value(np.random.randn()),
    "c": Value(np.random.randn()),
}

# Forward pass
output = tiny_mlp(inputs, params)
print(f"MLP output: {output.data:.5f}")

# Backward pass (backprop)
output.backward()

# Print gradients w.r.t. all parameters and inputs
print("\nGradients:")
for k, v in params.items():
    print(f"doutput/d{k} = {v.grad:.5f}")
for i, inp in enumerate(inputs):
    print(f"doutput/dx{i} = {inp.grad:.5f}")
