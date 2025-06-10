import torch

M3: torch.Tensor = torch.rand(3 * 3, dtype=torch.float32).reshape(3, 3)
M4: torch.Tensor = torch.rand(3 * 3, dtype=torch.float32).reshape(3, 3)

M34_sum: torch.Tensor = M3 + M4
M34_product: torch.Tensor = M3 * M4
M34_matmul: torch.Tensor = M3 @ M4

print("M3:\n", M3)
print("M4:\n", M4)
print("M34_sum:\n", M34_sum)
print("M34_product:\n", M34_product)
print("M34_matmul:\n", M34_matmul)
