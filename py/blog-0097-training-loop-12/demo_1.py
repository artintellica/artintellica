import torch
import matplotlib.pyplot as plt

# Make the output reproducible
torch.manual_seed(42)

N = 100  # Number of data points
X = torch.linspace(0, 1, N).unsqueeze(1)  # Shape: (N, 1)
true_w = torch.tensor([[2.0]])
true_b = torch.tensor([0.5])
y = X @ true_w + true_b + 0.1 * torch.randn(N, 1)  # Add some noise

# Initialize weight and bias (will be updated!)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = torch.Tensor([ 0.1 ])
num_epochs = 400

losses = []

for epoch in range(num_epochs):
    # 1. Forward pass: compute prediction
    y_pred = X @ w + b  # (N, 1)
    
    # 2. Compute mean squared error loss
    loss = ((y - y_pred)**2).mean()
    losses.append(loss.item())
    
    # 3. Backward pass: compute gradients
    loss.backward()
    
    # 4. Update parameters manually
    with torch.no_grad():
        w -= learning_rate * w.grad if w.grad is not None else 0
        b -= learning_rate * b.grad if b.grad is not None else 0

    # 5. Zero gradients for next iteration
    w.grad.zero_() if w.grad is not None else None
    b.grad.zero_() if b.grad is not None else None
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}: loss = {loss.item():.4f} w = {w.item():.4f} b = {b.item():.4f}")

# print shapes
print("X shape:", X.shape)
print("y shape:", y.shape)
# Plot original data and fitted line
plt.scatter(X.numpy().flatten(), y.numpy().flatten(), label="Data")
plt.plot(X.numpy().flatten(), (X @ w + b).detach().numpy().flatten(), 'r-', label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()

# Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss over Training")
plt.show()
