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

# Initialize parameters
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.04)

losses2 = []
for epoch in range(80):
    y_pred = w * x + b
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(loss.item())

print(f"Learned parameters: w={w.item():.2f}, b={b.item():.2f}")
plt.plot(losses2)
plt.title("Training Loss (With PyTorch Optimizer)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.grid(True)
plt.show()
