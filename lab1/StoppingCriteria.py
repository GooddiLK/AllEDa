# В этом файле содержатся критерии остановки
# GradientDescent требует, чтобы объект критерия остановки обладал мог быть вызван как функция (или был функцией),
# принимал объект типа GradientDescent и возвращал True при необходимости остановить алгоритм

import numpy as np


# Ограничение на кол-во итераций
class Iterations:
    def __init__(self, number):
        self.number = number

    def __call__(self, gd, point):
        return len(gd.history()) >= self.number


# Композиция Iterations с любым другим критерием остановки
class IterationsPlus:
    def __init__(self, number, plus):
        self.iterations = Iterations(number)
        self.plus = plus

    def __call__(self, gd, point):
        return self.iterations(gd, point) or self.plus(gd, point)


# Остановка произойдет, когда расстояние между соседними точками станет меньше epsilon
class SequenceEps:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, point):
        return np.linalg.norm(np.subtract(gd.history()[-1], point)) < self.eps

class SequenceValueEps:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, point):
        return abs(gd.func(gd.history()[-1]) - gd.func(point)) < self.eps
