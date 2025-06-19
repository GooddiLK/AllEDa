from lab2.Utils2 import *
from lab1.OneDimensional import *
import numpy as np

# принимает функцию,
# принимает q и c1 для функции ищущей следующий подходящий alpha для метода Армихо
class BFGS:
    def __init__(self, func, alpha_0, q, eps, c1, dem=2):
        variables = sp.symbols('x y')
        self.grad = compute_gradient(func, variables)
        self.bctrk = Backtracking(alpha_0, q, eps, c1)
        self.hess = np.eye(dem, dtype=np.longdouble)

    def learning_rate(self, gd):
        x = gd.history()[-1]
        x = np.array(x, dtype=np.longdouble)
        cur_grad = self.grad(*x)
        gd.vector = -self.hess @ cur_grad

        a = self.bctrk.calculate_alpha(gd, x)

        s = np.array(a * gd.vector, dtype=np.longdouble)
        x_new = np.array(x + s, dtype=np.longdouble)
        delta_grad = np.array(self.grad(*x_new), dtype=np.longdouble) - cur_grad

        rho = np.longdouble(1.0) / np.dot(delta_grad, s)
        i = np.eye(len(s), dtype=np.longdouble)
        a1 = i - rho * s[:, np.newaxis] * delta_grad[np.newaxis, :]
        a2 = i - rho * delta_grad[:, np.newaxis] * s[np.newaxis, :]

        self.hess = np.dot(a1, np.dot(self.hess, a2)) + (rho * s[:, np.newaxis] * s[np.newaxis, :])
        # TODO: Версия newtoneMethodAdding: self.hess = np.dot(a1, np.dot(self.hess, a2)) + (self.hess * s[:, np.newaxis] * s[np.newaxis, :])

        return a