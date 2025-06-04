import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a synthetic 4D tensor representing a batch of RGB images
# Shape: (2, 3, 64, 64) - (batch_size, channels, height, width)
batch_size = 2
channels = 3  # RGB
height = 64
width = 64
image_tensor = torch.randn(batch_size, channels, height, width)

# Print the shape before transposition
print(
    "Original Tensor Shape (batch_size, channels, height, width):", image_tensor.shape
)

# Transpose the dimensions to (2, 64, 64, 3) using permute
# Reorder dimensions from (batch_size, channels, height, width) to (batch_size, height, width, channels)
transposed_tensor = image_tensor.permute(0, 2, 3, 1)

# Print the shape after transposition
print(
    "Transposed Tensor Shape (batch_size, height, width, channels):",
    transposed_tensor.shape,
)

# Verify that the number of elements remains the same
print("\nNumber of elements in original tensor:", image_tensor.numel())
print("Number of elements in transposed tensor:", transposed_tensor.numel())
