import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(1e-6, 1 - 1e-6, 200)
bce_y1 = -np.log(p)  # when y = 1
bce_y0 = -np.log(1 - p)  # when y = 0

plt.plot(p, bce_y1, label="y=1")
plt.plot(p, bce_y0, label="y=0")
plt.xlabel("Predicted probability ($\\hat{y}$)")
plt.ylabel("BCE Loss")
plt.title("Binary Cross Entropy as a Function of $\\hat{y}$")
plt.ylim(0, 6)
plt.legend()
plt.grid(True)
plt.show()
