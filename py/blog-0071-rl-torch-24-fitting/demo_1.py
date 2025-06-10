import torch
import matplotlib.pyplot as plt

# True parameters
w_true = 2.5
b_true = -1.7
N = 120
torch.manual_seed(42)

# Generate random x and noisy y
x = torch.linspace(-3, 3, N)
y = w_true * x + b_true + 0.9 * torch.randn(N)

plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, w_true * x + b_true, 'k--', label='True line')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.title("Synthetic Linear Data"); plt.grid(True)
plt.show()

