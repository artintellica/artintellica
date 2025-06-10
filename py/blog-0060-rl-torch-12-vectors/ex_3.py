import torch

v: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print("v:", v)  # Tensor data
print("v first element:", v[0])  # Access first element
print("v last element:", v[-1])  # Access last element

# Mean
v_mean: torch.Tensor = v.mean()
# round to 2 decimal places
v_mean_rounded: torch.Tensor = torch.round(v_mean * 100) / 100
print("Mean of v:", v_mean_rounded.item())

# Sum
v_sum: torch.Tensor = v.sum()
# round to 2 decimal places
v_sum_rounded: torch.Tensor = torch.round(v_sum * 100) / 100
print("Sum of v:", v_sum_rounded.item())

# Standard deviation
v_std: torch.Tensor = v.std(unbiased=False)  # Match numpy's normalization
# round to 2 decimal places
v_std_rounded: torch.Tensor = torch.round(v_std * 100) / 100
print("Standard deviation of v:", v_std_rounded.item())
