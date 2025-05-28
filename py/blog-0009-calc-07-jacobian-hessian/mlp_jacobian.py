# calc-07-jacobian-hessian/mlp_jacobian.py
import torch

torch.set_default_dtype(torch.float64)


def mlp(x, W, b):
    return torch.tanh(W @ x + b)  # shape [2]


# random weights, biases, batch of points
W = torch.tensor([[1.0, -0.7], [0.4, 0.9]], requires_grad=True)
b = torch.tensor([0.5, -0.3], requires_grad=True)
xs = [
    torch.tensor([1.2, -0.8], requires_grad=True),
    torch.tensor([0.3, 2.0], requires_grad=True),
]

for i, x in enumerate(xs):
    y = mlp(x, W, b)
    J = torch.zeros((2, 2))
    for j in range(2):  # output dim
        grad = torch.autograd.grad(y[j], x, retain_graph=True, create_graph=True)[0]
        J[j] = grad
    print(f"\nJacobian at input {i}:")
    print(J.detach().numpy())
