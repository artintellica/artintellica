import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS backend is available!")
    mps_device = torch.device("mps")
    # Create a tensor on the GPU
    x = torch.ones(1, device=mps_device)
    print(f"Tensor on device: {x}")
else:
    print("MPS device not found.")
