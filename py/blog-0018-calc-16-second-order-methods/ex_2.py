import numpy as np

# Quadratic parameters
A = np.array([[5, 1], [1, 3]])
b = np.array([-4, 2])


# Quadratic function, grad, and Hessian
def f(x):
    return 0.5 * x @ A @ x + b @ x


def grad_f(x):
    return A @ x + b


def hess_f(x):
    return A


# Initial point (can be anything)
x0 = np.array([2.5, -2.0])

# Compute gradient and Hessian at x0
g = grad_f(x0)
H = hess_f(x0)

# Newton step: x_new = x0 - H^{-1} g
x_new = x0 - np.linalg.solve(H, g)

# Optimum (minimizer): x* = -A^{-1}b
x_star = -np.linalg.solve(A, b)

print(f"Initial x0:         {x0}")
print(f"Gradient at x0:     {g}")
print(f"Newton step x_new:  {x_new}")
print(f"Optimum x_star:     {x_star}")

print("\nDifference Newton step vs optimum: ", np.linalg.norm(x_new - x_star))

# For confirmation, show that the gradient at x_new is zero:
print("Gradient at x_new:", grad_f(x_new))
