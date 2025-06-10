import torch
import matplotlib.pyplot as plt


# - Create any $3 \times 5$ matrix with sequential values.
# - Plot the matrix and its transpose side-by-side using `imshow` and print their
#   shapes.
M: torch.Tensor = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5)
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
# Print shapes
print("Shape of M:", M.shape)
print("Shape of M.T:", M.T.shape)
