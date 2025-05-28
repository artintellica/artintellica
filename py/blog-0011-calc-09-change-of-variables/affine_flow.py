# calc-09-change-of-vars/affine_flow.py
import torch

torch.set_default_dtype(torch.float64)

# --- Parameters
A = torch.tensor([[2.0, 0.3], [0.1, 1.5]], requires_grad=True)
b = torch.tensor([1.0, -2.0], requires_grad=True)


# --- Inverse affine
def f(z):
    return A @ z + b


def f_inv(x):
    return torch.linalg.solve(A, x - b)


# --- Jacobian determinant (analytic, since affine)
def log_det_jacobian():
    return torch.logdet(A)


# --- Log-likelihood of x under the flow
def log_prob_x(x):
    z = f_inv(x)
    logpz = -0.5 * torch.dot(z, z) - torch.log(
        torch.tensor(2 * torch.pi)
    )  # standard 2D normal
    logdet = log_det_jacobian()
    return logpz - logdet


# --- Test
x = torch.tensor([2.0, 0.0], requires_grad=True)
print("x:", x.detach().numpy())
print("f⁻¹(x):", f_inv(x).detach().numpy())
print("log|det Jacobian|:", log_det_jacobian().item())
print("log-prob(x):", log_prob_x(x).item())


def autograd_logdet(x):
    x = x.clone().detach().requires_grad_(True)
    # z = f_inv(x)
    J = torch.autograd.functional.jacobian(f_inv, x)
    return torch.log(torch.abs(torch.det(J)))


print("Autograd log|det J|:", autograd_logdet(x).item())
