# calc-06-gradients/autograd_xy2.py
import torch

torch.set_default_dtype(torch.float64)


w = torch.tensor([0.7, -1.2], requires_grad=True)
x = torch.tensor([1.0, 2.0])
loss = (w @ x) ** 2  # scalar
loss.backward()
print("gradient wrt weights =", w.grad)
