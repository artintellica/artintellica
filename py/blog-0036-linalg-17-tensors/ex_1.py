import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create tensors of ranks 0 through 4 with different shapes
tensor_0d = torch.tensor(7.5)  # Rank 0 (scalar)
tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Rank 1 (vector of size 5)
tensor_2d = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
)  # Rank 2 (matrix of size 3x4)
tensor_3d = torch.randn(2, 3, 2)  # Rank 3 (3D tensor of size 2x3x2)
tensor_4d = torch.randn(2, 2, 3, 4)  # Rank 4 (4D tensor of size 2x2x3x4)

# Print shapes and number of elements for each tensor
print("Rank 0 Tensor (Scalar):")
print("Shape:", tensor_0d.shape)
print("Number of elements:", tensor_0d.numel())
print("Value:", tensor_0d)
print()

print("Rank 1 Tensor (Vector):")
print("Shape:", tensor_1d.shape)
print("Number of elements:", tensor_1d.numel())
print("Value:", tensor_1d)
print()

print("Rank 2 Tensor (Matrix):")
print("Shape:", tensor_2d.shape)
print("Number of elements:", tensor_2d.numel())
print("Value:", tensor_2d)
print()

print("Rank 3 Tensor (3D):")
print("Shape:", tensor_3d.shape)
print("Number of elements:", tensor_3d.numel())
print("Value:", tensor_3d)
print()

print("Rank 4 Tensor (4D):")
print("Shape:", tensor_4d.shape)
print("Number of elements:", tensor_4d.numel())
print("Value:", tensor_4d)
