# calc-12-ode/spiral_ode_numpy.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = -0.3
beta = 1.1


def f(t, h):
    x, y = h
    dx = alpha * x - beta * y
    dy = beta * x + alpha * y
    return [dx, dy]


t_span = [0, 8]
h0 = [2, 0.5]
sol = solve_ivp(f, t_span, h0, t_eval=np.linspace(*t_span, 400))

plt.plot(sol.y[0], sol.y[1], label="trajectory")
plt.title("Spiral ODE trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.axis("equal")
plt.show()
