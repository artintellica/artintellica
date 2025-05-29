# import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset (download if not already present)
mnist_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Get a single sample (image and label)
image, label = mnist_dataset[0]

# Flatten the image into a vector
# Image shape is (1, 28, 28), flatten to (784,)
feature_vector = image.flatten()

# Print dimension of the feature vector
print("Dimension of feature vector:", feature_vector.shape[0])

# Find the first 10 non-zero elements
non_zero_elements = []
non_zero_indices = []

for idx, value in enumerate(feature_vector):
    if value != 0:
        non_zero_elements.append(value.item())  # Convert tensor to scalar
        non_zero_indices.append(idx)
        if len(non_zero_elements) == 10:
            break

# Check if we found at least 10 non-zero elements
if len(non_zero_elements) < 10:
    raise ValueError(
        f"Unable to find 10 non-zero elements in the feature vector. Found only {len(non_zero_elements)} non-zero elements."
    )

# Convert to NumPy for printing and plotting
non_zero_elements = np.array(non_zero_elements)
non_zero_indices = np.array(non_zero_indices)

# Print the non-zero elements and their indices
print("First 10 non-zero elements:", non_zero_elements)
print("Corresponding indices:", non_zero_indices)

# Create bar plot for the first 10 non-zero elements
plt.figure(figsize=(8, 4))
plt.bar(non_zero_indices, non_zero_elements, color="blue", edgecolor="black")
plt.xlabel("Index in Feature Vector")
plt.ylabel("Pixel Intensity")
plt.title(f"First 10 Non-Zero Elements of MNIST Feature Vector (Digit: {label})")
plt.grid(True, axis="y")
plt.ylim(0, 1)  # Set y-axis limit to match pixel intensity range [0, 1]
plt.show()

# Optional: Visualize the full image for reference
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"MNIST Image (Digit: {label})")
plt.show()
