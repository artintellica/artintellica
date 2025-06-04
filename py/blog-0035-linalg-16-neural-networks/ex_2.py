import torch.nn as nn


# Define a neural network with two hidden layers in PyTorch
class TwoLayerNN(nn.Module):
    def __init__(self, input_size=3, hidden1_size=8, hidden2_size=4, output_size=2):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)  # Input to first hidden layer
        self.relu1 = nn.ReLU()  # Activation after first hidden layer
        self.layer2 = nn.Linear(
            hidden1_size, hidden2_size
        )  # First to second hidden layer
        self.relu2 = nn.ReLU()  # Activation after second hidden layer
        self.layer3 = nn.Linear(
            hidden2_size, output_size
        )  # Second hidden to output layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


# Instantiate the model
model = TwoLayerNN()
print("Model architecture:")
print(model)

# Print the shape of each weight matrix and bias vector
print("\nParameter shapes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
