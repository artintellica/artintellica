import torch

a: torch.Tensor = torch.tensor([2, 3, 4, 5], dtype=torch.float32)
b: torch.Tensor = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

# manual dot product
manual_dot: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)
for i in range(len(a)):
    manual_dot += a[i] * b[i]

# using pytorch's built-in dot product
builtin_dot: torch.Tensor = torch.dot(a, b)

print("a:", a)
print("b:", b)
print("Manual dot product (a · b):", manual_dot.item())
print("Built-in dot product (a · b):", builtin_dot.item())
