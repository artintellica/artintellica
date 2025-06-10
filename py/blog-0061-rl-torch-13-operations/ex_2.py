import torch

a: torch.Tensor = torch.tensor([2, 3, 4, 5], dtype=torch.float32)
c: torch.Tensor = torch.tensor(3, dtype=torch.float32)

# manual multiply with a loop
manual_product: torch.Tensor = torch.zeros_like(a)
for i in range(len(a)):
    manual_product[i] = a[i] * c

# using pytorch's built-in multiplication
built_in_product: torch.Tensor = a * c

print("a:", a)
print("c:", c)
print("Manual product (a * c):", manual_product)
print("Built-in product (a * c):", built_in_product)
