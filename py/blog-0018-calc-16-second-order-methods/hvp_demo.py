# calc-16-hessian-vector/hvp_demo.py
import torch

A = torch.tensor([[5.0, 1.0], [1.0, 3.0]], requires_grad=False)
b = torch.tensor([-4.0, 2.0], requires_grad=False)
x = torch.tensor([1.5, -1.2], requires_grad=True)


def f(x):
    return 0.5 * x @ A @ x + b @ x


fval = f(x)
grad = torch.autograd.grad(fval, x, create_graph=True)[0]
v = torch.tensor([0.6, -1.0])
# Compute Hessian-vector product
hvp = torch.autograd.grad(grad @ v, x)[0]
print("Hessian-vector product Hv:", hvp.numpy())
# Check against explicit H @ v
H = A.numpy()
print("Direct H @ v:", H @ v.numpy())
