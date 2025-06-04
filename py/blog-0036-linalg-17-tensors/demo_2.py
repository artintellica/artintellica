import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Broadcasting example
tensor_a = torch.tensor([1.0, 2.0, 3.0])  # Shape: (3,)
tensor_b = torch.tensor([[1.0], [2.0]])  # Shape: (2, 1)
result_broadcast = tensor_a + tensor_b  # Broadcasting: (2, 3)

print("\nBroadcasting result (tensor_a + tensor_b):", result_broadcast)
print("Result shape:", result_broadcast.shape)

# Shape tricks: Reshaping and unsqueezing
tensor_c = torch.randn(4, 3)  # Shape: (4, 3)
tensor_reshaped = tensor_c.reshape(2, 6)  # Reshape to (2, 6)
tensor_unsqueezed = tensor_c.unsqueeze(0)  # Add dimension at index 0, Shape: (1, 4, 3)

print("\nOriginal tensor shape:", tensor_c.shape)
print("Reshaped tensor shape:", tensor_reshaped.shape)
print("Unsqueezed tensor shape:", tensor_unsqueezed.shape)
