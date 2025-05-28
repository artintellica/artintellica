"""
exercise_1_affine_jacobian_check.py
-------------------------------------------------
For the affine flow f(z) = A z + b,
use autograd to compute the full Jacobian at random z,
and check that it matches A.
"""

import torch

torch.set_default_dtype(torch.float64)

# --- Parameters (choose any invertible A)
A = torch.tensor([[2.0, 0.3], [0.1, 1.5]], requires_grad=True)
b = torch.tensor([1.0, -2.0], requires_grad=True)


def f(z):
    return A @ z + b


# Test at several random z
for i in range(5):
    z = torch.randn(2, requires_grad=True)
    # Compute full Jacobian: shape [2, 2]
    J = torch.autograd.functional.jacobian(f, z)
    print(f"\nRandom z = {z.detach().numpy()}")
    print("Autograd Jacobian:\n", J.detach().numpy())
    print("A:\n", A.detach().numpy())
    print("Difference:\n", (J - A).detach().numpy())
    assert torch.allclose(J, A, atol=1e-10)
print("\n✓ All checked Jacobians match A (to 1e-10)")
