from __future__ import print_function

import torch

# print("test")
x = torch.empty(5, 3)
print(type(x))
# print(x)
x: torch.Tensor = torch.randn_like(x, dtype=torch.float)  # override dtype!
# print(x)  # result has the same size
# print(x.size)    # <built-in method size of Tensor object at 0x114d5c168>
# print(x.size())  # torch.Size([5, 3])
y: torch.Tensor = torch.rand(5, 3)
# print(y)
# print(x+y)
y.add_(x)
print(y)
y.add_(1)
print(y)
# print(x[:, 1])  # 各行tensorの2列目をベクトルとして表示

x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())
# print(x)
# print(y)
# print(z)
