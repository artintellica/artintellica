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


# --- Reverse-mode AD (backprop)
x = Value(1.0)
y = Value(2.0)
z = Value(3.0)
f = x + y + z
f.backward()
print("Reverse-mode AD (backprop):")
print(f"df/dx = {x.grad:.2f}, df/dy = {y.grad:.2f}, df/dz = {z.grad:.2f}")


# --- Forward-mode AD (manual Jacobian seeds)
def forward_jacobian(seed_x, seed_y, seed_z):
    # Manually propagate derivatives forward
    # f(x, y, z) = x + y + z
    df = seed_x * 1 + seed_y * 1 + seed_z * 1  # since ∂f/∂x = ∂f/∂y = ∂f/∂z = 1
    return df


print("\nForward-mode AD (manual):")
print("df/dx = ", forward_jacobian(1, 0, 0))
print("df/dy = ", forward_jacobian(0, 1, 0))
print("df/dz = ", forward_jacobian(0, 0, 1))

# --- Numerical check for completeness
eps = 1e-8


# f_num = lambda x0, y0, z0: x0 + y0 + z0
def f_num(x0, y0, z0):
    return x0 + y0 + z0


dfdx_num = (f_num(1.0 + eps, 2.0, 3.0) - f_num(1.0 - eps, 2.0, 3.0)) / (2 * eps)
dfdy_num = (f_num(1.0, 2.0 + eps, 3.0) - f_num(1.0, 2.0 - eps, 3.0)) / (2 * eps)
dfdz_num = (f_num(1.0, 2.0, 3.0 + eps) - f_num(1.0, 2.0, 3.0 - eps)) / (2 * eps)
print("\nNumerical check:")
print(f"df/dx = {dfdx_num:.2f}, df/dy = {dfdy_num:.2f}, df/dz = {dfdz_num:.2f}")
