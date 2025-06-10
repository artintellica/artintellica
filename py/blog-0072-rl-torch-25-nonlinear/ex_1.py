import torch
import matplotlib.pyplot as plt

# ### **Exercise 1:** Generate Polynomial Data with Noise

# - True function: $y_{\text{true}} = -0.5x^3 + 1.2x^2 - 0.7x + 4$.
# - Generate $N=150$ points in $x$ from $-2.5$ to $2.5$.
# - $y = y_{\text{true}} + \text{Gaussian noise (std=1.2)}$.
# - Plot $x$ and $y$ with the true curve.
w_true = -0.5
b_true = 1.2
N = 150
torch.manual_seed(42)
x = torch.linspace(-2.5, 2.5, N)
y_true = w_true * x**3 + b_true * x**2 - 0.7 * x + 4
y = y_true + 1.2 * torch.randn(N)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, y_true, 'k--', label='True curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Synthetic Polynomial Data with Noise")
plt.grid(True)
plt.show()
