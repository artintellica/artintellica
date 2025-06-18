
import torch

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA GPU!")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple M1/M2 GPU (MPS)!")
        return torch.device("mps")
    else:
        print("Using CPU.")
        return torch.device("cpu")
