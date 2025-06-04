import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a 3D tensor of shape (2, 3, 4) with random values
tensor_3d = torch.randn(2, 3, 4)
print("Original 3D Tensor:")
print("Shape:", tensor_3d.shape)
print("Value:")
print(tensor_3d)

# Reshape to a 2D tensor of shape (6, 4)
tensor_2d = tensor_3d.reshape(6, 4)
print("\nReshaped to 2D Tensor:")
print("Shape:", tensor_2d.shape)
print("Value:")
print(tensor_2d)

# Reshape to a 1D tensor of size 24
tensor_1d = tensor_3d.reshape(24)
print("\nReshaped to 1D Tensor:")
print("Shape:", tensor_1d.shape)
print("Value:")
print(tensor_1d)
