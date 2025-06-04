import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create two tensors with shapes (3, 1) and (1, 4)
tensor_a = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: (3, 1)
tensor_b = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Shape: (1, 4)

print("Tensor A (shape:", tensor_a.shape, "):")
print(tensor_a)
print("\nTensor B (shape:", tensor_b.shape, "):")
print(tensor_b)

# Perform addition using broadcasting
result_add = tensor_a + tensor_b
print("\nResult of Addition (A + B) using broadcasting:")
print("Shape:", result_add.shape)
print("Values:")
print(result_add)

# Perform multiplication using broadcasting
result_mul = tensor_a * tensor_b
print("\nResult of Multiplication (A * B) using broadcasting:")
print("Shape:", result_mul.shape)
print("Values:")
print(result_mul)

# Explanation of broadcasting
print("\nExplanation of Broadcasting:")
print(
    "Broadcasting allows operations between tensors of different shapes by automatically expanding their dimensions."
)
print(
    "Tensor A (3, 1) is expanded along the second dimension to (3, 4) by repeating its column 4 times."
)
print(
    "Tensor B (1, 4) is expanded along the first dimension to (3, 4) by repeating its row 3 times."
)
print(
    "After expansion, both tensors are of shape (3, 4), enabling element-wise addition and multiplication."
)
