import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def make_toy_data(n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    # Data: input x, labels y = sin(x)
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n).unsqueeze(1)
    y = torch.sin(x)
    return x, y

class SimpleNet(torch.nn.Module):
    def __init__(self, use_activation: bool):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.use_activation = use_activation
        self.fc2 = torch.nn.Linear(10, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_activation:
            x = torch.tanh(x)  # Try also with torch.tanh(x)
        x = self.fc2(x)
        return x

def train_model(use_activation: bool, epochs: int = 1000) -> torch.nn.Module:
    x, y = make_toy_data()
    model = SimpleNet(use_activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def plot_fit_with_and_without_activation():
    x, y = make_toy_data()
    models = {
        "No Activation (Linear Only)": train_model(use_activation=False, epochs=1000),
        "With ReLU Activation": train_model(use_activation=True, epochs=1000),
    }

    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy(), label="True sin(x)", color="black", linewidth=2)
    for label, model in models.items():
        y_pred = model(x).detach()
        plt.plot(x.numpy(), y_pred.numpy(), label=label)

    plt.title("Linear v.s. Non-linear (ReLU) Neural Network Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_fit_with_and_without_activation()
