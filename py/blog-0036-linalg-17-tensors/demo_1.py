import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create tensors of different ranks and shapes
scalar = torch.tensor(5.0)  # 0D tensor (scalar)
vector = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor (vector)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor (matrix)
tensor_3d = torch.randn(2, 3, 4)  # 3D tensor (random values)

print("Scalar (0D):", scalar, "Shape:", scalar.shape)
print("Vector (1D):", vector, "Shape:", vector.shape)
print("Matrix (2D):", matrix, "Shape:", matrix.shape)
print("3D Tensor:", tensor_3d, "Shape:", tensor_3d.shape)

# Basic operations
sum_tensor = vector + 2.0  # Element-wise addition with scalar
product_tensor = vector * matrix[:, 0]  # Element-wise multiplication (broadcasting)
matmul_result = torch.matmul(matrix, matrix)  # Matrix multiplication

print("\nElement-wise addition (vector + scalar):", sum_tensor)
print("Element-wise multiplication (vector * matrix column):", product_tensor)
print("Matrix multiplication (matrix @ matrix):", matmul_result)
