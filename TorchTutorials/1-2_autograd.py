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

# print("x.grad = " + str(x.grad))
# We have that o = (1/4)∑_i zi
# , zi = 3(xi+2)^2
#  and zi∣∣xi=1 =27
# . Therefore, ∂o/∂xi = 3/2(xi+2)
# , hence ∂o/∂xi∣∣xi=1 = 9/2 = 4.5
# .

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print("y = " + str(y))

# Now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly,
# but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("x.grad = " + str(x.grad))

# You can also stop autograd from tracking history on Tensors with .requires_grad=True
# by wrapping the code block in with torch.no_grad():

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():   # excecute deinit().temporary no grad.
    print((x ** 2).requires_grad)

print((x ** 2).requires_grad)
