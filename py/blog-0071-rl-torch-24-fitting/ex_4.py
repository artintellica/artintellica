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

# ### **Exercise 3:** Use PyTorch’s Autograd and Optimizer to Fit a Line

# - Use `torch.optim.SGD` or `torch.optim.Adam`.
# - Train for 100 epochs.
# - Plot loss vs. epoch and print the learned $w, b$.
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.04)
losses = []
for epoch in range(100):
    y_pred = w * x + b  # Linear model
    loss = torch.nn.functional.mse_loss(y_pred, y)  # MSE
    loss.backward()
    
    optimizer.step()  # Update parameters
    optimizer.zero_grad()  # Zero gradients
    
    losses.append(loss.item())
    
    if epoch % 20 == 0 or epoch == 99:
        print(f"Epoch {epoch:2d}: w={w.item():.2f}, b={b.item():.2f}, loss={loss.item():.3f}")

# ### **Exercise 4:** Plot Predictions vs. Ground Truth and Compute $R^2$ Score

# - Plot the original data, the true line, and the model’s predictions.
# - Compute and print the $R^2$ score for your fitted model.
with torch.no_grad():
    y_fit = w * x + b
plt.scatter(x, y, label="Data", alpha=0.6)
plt.plot(x, w_true * x + b_true, "k--", label="True line")
plt.plot(x, y_fit, "r-", label="Fitted line")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Fit vs Ground Truth")
plt.grid(True)
plt.show()
