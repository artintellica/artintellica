import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.1])  # Linear with small noise

w_vals = np.linspace(0, 3, 100)
loss_vals = [np.mean((w * x - y)**2) for w in w_vals]

plt.plot(w_vals, loss_vals)
plt.xlabel("w")
plt.ylabel("MSE Loss")
plt.title("Loss Surface for Linear Model y = w*x")
plt.grid(True)
plt.show()
