import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


def trace_graph(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                if isinstance(child, Value):
                    edges.add((child, v))
                    build(child)

    build(root)
    return nodes, edges


def plot_graph(root):
    nodes, edges = trace_graph(root)
    G = nx.DiGraph()
    labels = {}
    for n in nodes:
        label = f"{n._op or 'input'}\n{n.data:.2f}\ngrad={n.grad:.2f}"
        G.add_node(id(n), label=label)
        labels[id(n)] = label
    for src, tgt in edges:
        G.add_edge(id(src), id(tgt))
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=False, arrows=True, node_color="lightblue", node_size=1500
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Computation Graph")
    plt.show()


a = Value(2.0)
b = Value(-3.0)
c = a * b + a**2
d = c.tanh()
d.backward()
plot_graph(d)
