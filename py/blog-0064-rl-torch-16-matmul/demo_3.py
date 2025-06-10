import torch
import matplotlib.pyplot as plt

# Visualize data of a matrix and its transpose
M: torch.Tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(M, cmap="viridis", aspect="auto")
axs[0].set_title("Original M\nshape={}".format(M.shape))
axs[0].set_xlabel("Columns")
axs[0].set_ylabel("Rows")
axs[1].imshow(M.T, cmap="viridis", aspect="auto")
axs[1].set_title("Transposed M\nshape={}".format(M.T.shape))
axs[1].set_xlabel("Columns")
axs[1].set_ylabel("Rows")
plt.tight_layout()
plt.show()
