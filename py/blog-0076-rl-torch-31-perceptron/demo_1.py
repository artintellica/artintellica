import torch

def perceptron_step(
    x: torch.Tensor,
    y: float,
    w: torch.Tensor,
    b: float,
    lr: float = 1.0
) -> tuple[torch.Tensor, float]:
    """
    x: input vector (d,)
    y: true label (0 or 1)
    w: weight vector (d,)
    b: bias (scalar)
    Returns updated w, b
    """
    # Prediction: if w^T x + b > 0: 1 else 0
    z: float = torch.dot(w, x).item() + b
    y_pred: float = 1.0 if z > 0 else 0.0
    if y != y_pred:
        # Update rule
        w = w + lr * (y - y_pred) * x
        b = b + lr * (y - y_pred)
    return w, b

# Simple test
w: torch.Tensor = torch.zeros(2)
b: float = 0.0
x: torch.Tensor = torch.tensor([1.0, -1.0])
y: float = 1.0
w, b = perceptron_step(x, y, w, b)
print("Updated w/b:", w, b)
