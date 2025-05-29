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

# Reshape to 14x56
image_reshaped = image_matrix.reshape(14, 56)

# Print shapes
print("Original image shape:", image_matrix.shape)
print("Reshaped image shape:", image_reshaped.shape)

# Visualize both images
plt.figure(figsize=(10, 4))

# Original 28x28 image
plt.subplot(1, 2, 1)
plt.imshow(image_matrix, cmap="gray")
plt.title(f"Original 28x28 MNIST Digit: {label}")
plt.colorbar(label="Pixel Intensity")

# Reshaped 14x56 image
plt.subplot(1, 2, 2)
plt.imshow(image_reshaped, cmap="gray")
plt.title(f"Reshaped 14x56 MNIST Digit: {label}")
plt.colorbar(label="Pixel Intensity")

plt.tight_layout()
plt.show()
