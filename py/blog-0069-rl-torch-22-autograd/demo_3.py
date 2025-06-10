import torch

w = torch.tensor(1.0, requires_grad=True)

for step in range(2):
    f = (w - 2) ** 2  # Simple loss
    f.backward()
    print(
        f"Step {step}: w.grad = {w.grad.item() if w.grad is not None else 'No gradient computed'}"
    )
    if w.grad is not None:
        w.data.zero_()  # Reset the data to zero
