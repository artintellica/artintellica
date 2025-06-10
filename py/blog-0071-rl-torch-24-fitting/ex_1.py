import torch
import matplotlib.pyplot as plt

# ### **Exercise 1:** Generate Synthetic Linear Data with Noise

# - True line: $w_\text{true} = 1.7$, $b_\text{true} = -0.3$.
# - Sample $N=100$ points for $x$ from -2 to 4.
# - $y = w_{\text{true}}x + b_{\text{true}} +$ noise (Gaussian, std=0.5).
# - Plot $x$ and $y$ with the true line.
w_true = 1.7
b_true = -0.3
N = 100
torch.manual_seed(42)
x = torch.linspace(-2, 4, N)
y = w_true * x + b_true + 0.5 * torch.randn(N)
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, w_true * x + b_true, 'k--', label='True line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Synthetic Linear Data with Noise")
plt.grid(True)
plt.show()
