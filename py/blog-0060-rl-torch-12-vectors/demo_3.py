import torch

# Scalar (0-dimensional tensor, holds a single value)
a: torch.Tensor = torch.tensor(7.5)
print("Scalar a:", a)
print("Shape (should be torch.Size([])):", a.shape)
print("Dimensions (should be 0):", a.dim())

# 1-D vector tensor
v: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Vector v:", v)
print("Shape:", v.shape)
print("Dimensions:", v.dim())

# Compute statistics
mean_v: torch.Tensor = v.mean()
sum_v: torch.Tensor = v.sum()
std_v: torch.Tensor = v.std(unbiased=False)  # Match numpy's normalization

print(f"Mean: {mean_v.item():.2f}")
print(f"Sum: {sum_v.item():.2f}")
print(f"Std: {std_v.item():.2f}")
