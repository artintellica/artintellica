import torch
import numpy as np
import matplotlib.pyplot as plt


def perceptron_step(
    x: torch.Tensor, y: float, w: torch.Tensor, b: float, lr: float = 1.0
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


# test the perceptron_step function
def test_perceptron_step():
    x = torch.tensor([1.0, 2.0])
    y = 1.0
    w = torch.tensor([0.5, -0.5])
    b = 0.0
    lr = 0.1

    updated_w, updated_b = perceptron_step(x, y, w, b, lr)
    print("Updated weights:", updated_w)
    print("Updated bias:", updated_b)


# Run the test
test_perceptron_step()
