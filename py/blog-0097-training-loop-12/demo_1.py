import torch
import matplotlib.pyplot as plt

# Make reproducible
torch.manual_seed(42)

true_w = 2.0  # True slope
true_b = -1.0  # True intercept

N = 100  # Number of data points

# x: Uniformly spaced between -1 and 1
x: torch.Tensor = torch.linspace(-1, 1, N).unsqueeze(1)  # Shape (N, 1)
# y: Linear relation plus some noise
y: torch.Tensor = true_w * x + true_b + 0.1 * torch.randn(N, 1)

plt.scatter(x.numpy(), y.numpy(), label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fake Linear Data")
plt.legend()
plt.show()
