import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Download & normalize
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
