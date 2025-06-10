import torch

v: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print("v:", v)  # Tensor data
print("v first element:", v[0])  # Access first element
print("v last element:", v[-1])  # Access last element

# Mean
v_mean: torch.Tensor = v.mean()
# round to 2 decimal places
print("Mean of v: {:.2f}".format(v_mean.item()))

# Sum
v_sum: torch.Tensor = v.sum()
# round to 2 decimal places
print("Sum of v: {:.2f}".format(v_sum.item()))

# Standard deviation
v_std: torch.Tensor = v.std(unbiased=False)  # Match numpy's normalization
# round to 2 decimal places
print("Standard deviation of v: {:.2f}".format(v_std.item()))
