import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a synthetic dataset with 3 features (100 samples)
n_samples = 100
input_features = 3
X = np.random.randn(n_samples, input_features) * 2  # Shape: (100, 3)
X = torch.FloatTensor(X)
print("Input Data Shape (batch_size, features):", X.shape)


# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.layer2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer

    def forward(self, x):
        print("\nForward Pass Shapes:")
        print("Input Shape:", x.shape)

        # First layer transformation
        x1 = self.layer1(x)
        print("After Layer 1 (Linear) Shape:", x1.shape)

        # Activation
        x2 = self.relu(x1)
        print("After ReLU Activation Shape:", x2.shape)

        # Second layer transformation
        x3 = self.layer2(x2)
        print("After Layer 2 (Linear) Shape:", x3.shape)

        return x3


# Instantiate the model
model = SimpleNN()
print("Model Architecture:")
print(model)

# Print the shapes of the weight tensors
print("\nWeight Tensor Shapes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Perform a forward pass to observe shape transformations
output = model(X)
print("\nFinal Output Shape:", output.shape)
