import matplotlib.pyplot as plt
import torch

# Generate synthetic data
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(X.size()) * 0.5

# Define a linear model
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot results
plt.scatter(X, y, label="Data")
plt.plot(X, model(X).detach(), color="red", label="Fit")
plt.legend()
plt.show()
