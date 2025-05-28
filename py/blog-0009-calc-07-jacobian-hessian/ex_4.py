"""
exercise_4_fastest_direction.py
-------------------------------------------------
For a tiny MLP with scalar output, compute the Hessian at a point,
find the eigenvector with largest-magnitude eigenvalue (|λ|max), and
plot f(param_vec + α*vmax) for α ∈ [−1,1] to show how fast the function
changes in that direction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# --- 1. Define tiny MLP
def net(flat_params, x):
    W1 = flat_params[0:4].view(2, 2)
    b1 = flat_params[4:6]
    W2 = flat_params[6:8].view(1, 2)
    b2 = flat_params[8:9]
    h = torch.tanh(W1 @ x + b1)
    return (W2 @ h + b2).squeeze()


# --- 2. Initialize parameters and input
param_vec = torch.randn(9, requires_grad=True)
x = torch.randn(2, requires_grad=False)  # input fixed


# --- 3. Compute output and Hessian wrt parameters
def compute_hessian(f, params):
    n = params.numel()
    (grad1,) = torch.autograd.grad(f, params, create_graph=True)
    hess = torch.zeros(n, n, dtype=params.dtype)
    for i in range(n):
        (grad2,) = torch.autograd.grad(grad1[i], params, retain_graph=True)
        hess[i] = grad2
    return hess.detach().numpy()


y = net(param_vec, x)
H = compute_hessian(y, param_vec)

# --- 4. Eigen-decomposition
eigvals, eigvecs = np.linalg.eigh(H)
imax = np.argmax(np.abs(eigvals))
vmax = eigvecs[:, imax]
lmax = eigvals[imax]
print(f"Largest-magnitude eigenvalue: {lmax:.3f}")
print("Eigenvector (normalized):", vmax)

# --- 5. Plot f(param_vec + α * vmax) for α ∈ [−1, 1]
alphas = np.linspace(-1, 1, 101)
fvals = []
with torch.no_grad():
    for a in alphas:
        w = torch.tensor(param_vec.detach().numpy() + a * vmax, dtype=param_vec.dtype)
        fvals.append(float(net(w, x)))

plt.figure(figsize=(6, 4))
plt.plot(alphas, fvals, label="f(param_vec + α·vₘₐₓ)")
plt.xlabel("α")
plt.ylabel("network output")
plt.title("Function value along fastest curvature direction")
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.legend()
plt.show()
