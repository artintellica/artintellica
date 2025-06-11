import torch
import matplotlib.pyplot as plt

# ### **Exercise 1:** Generate Binary Classification Data

# - Make two clouds in 2D using Gaussians with means $[0, 0]$ and $[3, 2]$ and a
#   shared covariance.
# - Stack them to form $N=100$ dataset and make integer labels $0$ and $1$.
# - Plot the dataset, color by class.
mean0 = torch.tensor([0.0, 0.0])
mean1 = torch.tensor([3.0, 2.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.2]])
L = torch.linalg.cholesky(cov)
N = 100
X0 = torch.randn(N // 2, 2) @ L.T + mean0
X1 = torch.randn(N // 2, 2) @ L.T + mean1
X = torch.cat([X0, X1], dim=0)
y = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)])

# ### **Exercise 2:** Implement Logistic Regression “From Scratch” (Sigmoid + BCE)

# - Add a bias column to data.
# - Initialize weights as zeros (with `requires_grad=True`).
# - Use the sigmoid and BCE formulas explicitly in your training loop.
# - Train for $1000$ epochs, plot the loss curve.
def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))
# Add bias feature for simplicity: [x1, x2, 1]
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # shape (N, 3)
w = torch.zeros(3, requires_grad=True)
lr = 0.05
losses = []
for epoch in range(1000):
    z = X_aug @ w  # Linear
    p = sigmoid(z)  # Probabilities
    # Numerical stabilization: clamp p
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    bce = (-y * torch.log(p) - (1 - y) * torch.log(1 - p)).mean()
    bce.backward()
    with torch.no_grad():
        w -= lr * w.grad if w.grad is not None else 0
    w.grad.zero_() if w.grad is not None else None
    losses.append(bce.item())
    if epoch % 200 == 0 or epoch == 999:
        print(f"Epoch {epoch}: BCE loss={bce.item():.3f}")
plt.plot(losses)
plt.title("Training Loss: Logistic Regression (Scratch)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
plt.show()
