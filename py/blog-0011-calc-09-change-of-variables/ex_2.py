"""
exercise_2_nonlinear_flow_jacobian.py
-------------------------------------------------
For f(z) = tanh(Az + b), compute the Jacobian and its log-determinant
at a random z, using autograd. Try with different A.
"""

import torch
import numpy as np

torch.set_default_dtype(torch.float64)


def f(z, A, b):
    return torch.tanh(A @ z + b)


# --- Try a few different A matrices
A_list = [
    torch.tensor([[1.0, 0.5], [0.2, 1.3]], dtype=torch.float64),
    torch.tensor([[2.0, -0.3], [0.8, 0.5]], dtype=torch.float64),
    torch.tensor([[0.5, 1.2], [-1.0, 2.0]], dtype=torch.float64),
]

b = torch.tensor([0.7, -1.5], dtype=torch.float64)

for idx, A in enumerate(A_list):
    print(f"\n--- Case {idx + 1}: ---")
    z = torch.randn(2, dtype=torch.float64, requires_grad=True)
    print("z =", z.detach().numpy())
    print("A =\n", A.detach().numpy())
    # Compute Jacobian via autograd
    J = torch.autograd.functional.jacobian(lambda z_: f(z_, A, b), z)
    print("Jacobian:\n", J.detach().numpy())
    # Log-determinant
    detJ = torch.det(J)
    print("determinant =", detJ.item())
    if detJ.item() != 0.0:
        logdet = torch.log(torch.abs(detJ))
        print("log|det J| =", logdet.item())
    else:
        print("determinant is zero (singular Jacobian)")

    # For curiosity, also print singular values
    svd = np.linalg.svd(J.detach().numpy(), compute_uv=False)
    print("Singular values:", svd)
