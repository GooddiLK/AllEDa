# В этом файле содержатся алгоритмы планирования шага
# GradientDescent требует, чтобы объект планирования шага обладал методом learning_rate,
# принимающим объект типа GradientDescent и возвращающим текущий шаг


import numpy as np


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


# Класс, реализующий метод Армихо
class Armijo:
    def __init__(self, alpha_0, q, eps, c1):
        self.bctrk = Backtracking(alpha_0, q, eps, c1)

    def learning_rate(self, gd):
        return self.bctrk.calculate_alpha(gd, gd.history()[-1])


# Класс, реализующий бинпоиск подходящей точки для алгоритма Вольфа
class BinarySearch:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, x_k, c1, c2, l, r):
        while r - l > self.eps:
            m = (l + r) / 2
            mp = gd.next_point(x_k, m)
            b1 = gd.func(mp) < l_alpha(gd, c1, x_k, m)
            b2 = np.sum(np.multiply(gd.grad(mp), gd.vector)) >= c2 * (np.linalg.norm(gd.grad(x_k))) ** 2
            if not b1:
                r = m
                continue
            if not b2:
                l = m
                continue
            return m
        return 0


# Класс, реализующий метод Вольфа
class Wolfe:
    def __init__(self, alpha_0, c1, c2, eps):
        if not (0 < c1 < c2 < 1):
            raise Exception("Incorrect parameters")
        self.c1 = c1
        self.c2 = c2
        self.alpha_0 = alpha_0
        self.eps = eps

    def learning_rate(self, gd):
        bs = BinarySearch(self.eps)
        return bs(gd, gd.history()[-1], self.c1, self.c2, 0, self.alpha_0)
