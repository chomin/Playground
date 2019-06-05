import numpy as np

x_size = 28*28
w = np.random.normal(0, 10, x_size)
# [4.8 1.0 -2.5 6.0 -3.0 ...

e1 = 0.1 * np.sign(np.random.normal(0, 10, x_size))
# [0.1 -0.1 -0.1 -0.1 0.1 ...
e2 = 0.1 * np.sign(w)
# [0.1 0.1 -0.1 0.1 -0.1 ...

print('w dot e1:', abs(np.dot(w.T, e1)))   # 0.510
print('w dot e2:', abs(np.dot(w.T, e2)))   # 733.0
