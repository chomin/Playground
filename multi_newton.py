# 多変数のニュートン法のサンプル
#
# https://qiita.com/utahkaA/items/45ebadd70c3020e8c320

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 13
plt.rcParams["figure.figsize"] = 16, 12


class MultiNewton(object):
    def __init__(self, f, dx0_f, dx1_f, grad_f, hesse):
        self.f = f
        self.dx0_f = dx0_f
        self.dx1_f = dx1_f
        self.grad_f = grad_f
        self.hesse = hesse
        self.hist = None

    def _compute_dx(self, bar_x):
        grad = self.grad_f(bar_x)
        hesse = self.hesse(bar_x)
        dx = np.linalg.solve(hesse, -grad)
        return dx

    def solve(self, init_x, n_iter=100, tol=0.01, step_width=1.0):
        self.hist = np.zeros(n_iter)
        bar_x = init_x
        x = 0.0
        for i in range(n_iter):
            dx = self._compute_dx(bar_x)
            # update
            x = bar_x + dx*step_width
            print("x = [{0:.2f} {1:.2f}]".format(x[0], x[1]))

            bar_x = x
            norm_dx = np.linalg.norm(dx)
            self.hist[i] += norm_dx
            if norm_dx < tol:
                self.hist = self.hist[:i]
                break
        return x


def _main():
    f = lambda x: x[0]**3 + x[1]**3 - 9*x[0]*x[1] + 27  # lambdaは無名関数のこと
    dx0_f = lambda x: 3*x[0]**2 - 9*x[1]
    dx1_f = lambda x: 3*x[1]**2 - 9*x[0]
    grad_f = lambda x: np.array([dx0_f(x), dx1_f(x)])
    hesse = lambda x: np.array([[6*x[0], -9], [-9, 6*x[1]]])

    init_x = np.array([10, 8])
    solver = MultiNewton(f, dx0_f, dx1_f, grad_f, hesse)
    res = solver.solve(init_x=init_x, n_iter=100, step_width=1.0)
    print("Solution is x = [{0:.2f} {1:.2f}]".format(res[0], res[1]))

    errors = solver.hist
    epochs = np.arange(0, errors.shape[0])

    plt.plot(epochs, errors)
    plt.tight_layout()
    plt.savefig('error_mult.png')


if __name__ == "__main__":
    _main()
