# PyTorch Tutorial
# DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ
# AUTOGRAD: AUTOMATIC DIFFERENTIATION
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

import torch

x = torch.ones(2, 2, requires_grad=True)  # x.requires_grad_(True)で後から設定可
# print(x)

y = x + 2   # yは計算結果なのでgrad_fnを持つ
# print(y)
# print(y.grad_fn)

z = y*y*3
out: torch.Tensor = z.mean()

# print(z, out)

out.backward()

print(x.grad)
# We have that o = (1/4)∑_i zi
# , zi = 3(xi+2)^2
#  and zi∣∣xi=1 =27
# . Therefore, ∂o/∂xi = 3/2(xi+2)
# , hence ∂o/∂xi∣∣xi=1 = 9/2 = 4.5
# .
