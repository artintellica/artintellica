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

# ### **Exercise 2:** Implement Linear Regression Training Loop from Scratch (Only Tensors!)
# - Randomly initialize $w, b$ (set `requires_grad=True`).
# - For 100 epochs: predict, compute loss (MSE), backward, update $w, b$ with a
#   learning rate (no optimizer object).
# - Zero grads after each update.
# - Plot the loss curve.
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.04
losses = []
for epoch in range(100):
    y_pred = w * x + b  # Linear model
    loss = ((y_pred - y) ** 2).mean()  # MSE
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad if w.grad is not None else 0
        b -= lr * b.grad if b.grad is not None else 0

    if w.grad is None or b.grad is None:
        break

    w.grad.zero_()
    b.grad.zero_()
    losses.append(loss.item())

    if epoch % 20 == 0 or epoch == 99:
        print(
            f"Epoch {epoch:2d}: w={w.item():.2f}, b={b.item():.2f}, loss={loss.item():.3f}"
        )
plt.plot(losses)
plt.title("Training Loss (From Scratch)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
