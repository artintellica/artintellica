import torchvision
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Get a single image
image, label = mnist_dataset[0]

# Convert to NumPy array (shape: 1, 28, 28 -> 28, 28)
image_matrix = image.squeeze().numpy()

# Print shape and sample elements
print("Image matrix shape:", image_matrix.shape)
print("Top-left 3x3 corner:\n", image_matrix[:3, :3])

# Visualize the image matrix
plt.figure(figsize=(5, 5))
plt.imshow(image_matrix, cmap="gray")
plt.title(f"MNIST Digit: {label}")
plt.colorbar(label="Pixel Intensity")
plt.show()


# Flatten the image matrix into a vector
image_vector = image_matrix.flatten()

# Print new shape
print("Flattened vector shape:", image_vector.shape)

# Reshape back to 28x28
image_matrix_reshaped = image_vector.reshape(28, 28)

# Verify shapes match
print("Reshaped matrix shape:", image_matrix_reshaped.shape)
