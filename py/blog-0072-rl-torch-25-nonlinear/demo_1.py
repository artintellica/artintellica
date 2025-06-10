import torch
import matplotlib.pyplot as plt

N = 120
torch.manual_seed(0)
x = torch.linspace(-3, 3, N)
# True relationship: cubic (degree 3) with noise
y_true = 0.4 * x**3 - x**2 + 0.5 * x + 2.0
y = y_true + 2.0 * torch.randn(N)

plt.scatter(x, y, alpha=0.5, label="Data (noisy)")
plt.plot(x, y_true, "k--", label="True curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Polynomial Data")
plt.legend()
plt.show()
