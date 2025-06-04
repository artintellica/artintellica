import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create two 3D tensors representing batches of matrices
batch_size = 5
tensor_a = torch.randn(
    batch_size, 3, 2
)  # Shape: (5, 3, 2) - batch of 5 matrices, each 3x2
tensor_b = torch.randn(
    batch_size, 2, 4
)  # Shape: (5, 2, 4) - batch of 5 matrices, each 2x4

print("Tensor A (shape:", tensor_a.shape, "):")
print(tensor_a)
print("\nTensor B (shape:", tensor_b.shape, "):")
print(tensor_b)

# Perform batch matrix multiplication using torch.bmm
result_bmm = torch.bmm(tensor_a, tensor_b)  # Shape should be (5, 3, 4)
print("\nResult of Batch Matrix Multiplication (torch.bmm):")
print("Shape:", result_bmm.shape)
print("Values:")
print(result_bmm)

# Explanation of the operation
print("\nExplanation of Batch Matrix Multiplication (torch.bmm):")
print("torch.bmm performs matrix multiplication on batches of matrices independently.")
print("Tensor A has shape (5, 3, 2), representing 5 matrices of size 3x2.")
print("Tensor B has shape (5, 2, 4), representing 5 matrices of size 2x4.")
print(
    "For each batch index (1 to 5), the corresponding 3x2 matrix from A is multiplied with the corresponding 2x4 matrix from B."
)
print(
    "This results in a 3x4 matrix for each batch index, leading to a final tensor of shape (5, 3, 4)."
)
print(
    "The operation is equivalent to performing 5 separate matrix multiplications, one for each pair of matrices in the batch."
)
