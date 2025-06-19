import numpy as np
from lab2.Utils import *


# Вычисление одной из частей условия Армихо
def l_alpha(gd, c1, x_k, alpha):
    return gd.func(x_k) - c1 * alpha * (np.linalg.norm(gd.grad(x_k))) ** 2


# Класс, реализующий метод backtracking
class Backtracking:
    def __init__(self, alpha_0, q, eps, c1):
        self.alpha_0 = alpha_0
        self.q = q
        self.eps = eps
        self.c1 = c1

    # Функция ищущая следующий подходящий alpha для метода Армихо
    def calculate_alpha(self, gd, x_k):
        alpha = self.alpha_0
        while True:
            if alpha < self.eps:
                return alpha
            next_point = gd.next_point(x_k, alpha)
            if gd.func(next_point) < l_alpha(gd, self.c1, x_k, alpha):
                return alpha
            else:
                alpha = self.q * alpha



# принимает функцию
# принимает q и c1 для функции ищущущей следующий подходящий alpha для метода Армихо
class BFGS:
    def __init__(self, func, alpha_0, q, eps, c1, dem=2):
        vars = sp.symbols('x y')
        self.grad = compute_gradient(func, vars)
        self.bctrk = Backtracking(alpha_0, q, eps, c1)
        self.hess = np.eye(dem)

    def learning_rate(self, gd):
        x = gd.history()[-1]
        x = np.array(x)
        cur_grad = self.grad(*x)
        gd.vector = -self.hess @ cur_grad

        a = self.bctrk.calculate_alpha(gd, x)

        s = a * gd.vector
        x_new = x + s
        delta_grad = np.array(self.grad(*x_new)) - cur_grad

        rho = 1.0 / np.dot(delta_grad, s)
        I = np.eye(len(s))
        A1 = I - rho * s[:, np.newaxis] * delta_grad[np.newaxis, :]
        A2 = I - rho * delta_grad[:, np.newaxis] * s[np.newaxis, :]

        self.hess = np.dot(A1, np.dot(self.hess, A2)) + ( s[:, np.newaxis] *
                                           s[np.newaxis, :])

        return a