# В этом файле содержится непосредственно алгоритм градиентного спуска

import numpy as np

from StoppingCriteria import IterationsPlus


class GradientDescent:
    def __init__(self, func, grad, learningRateCalculator, stoppingCriteria):
        self.__funcFunc__ = func
        self.__gradFunc__ = grad
        # Calculator должен обладать методом принимающим экземпляр этого класса и возвращающим следующий learning_rate
        self.learningRateCalculator = learningRateCalculator
        # stoppingCriteria должен обладать методом принимающим экземпляр этого класса и последнюю вычисленную точку и возвращающим True,
        # если необходимо закончить вычисление
        self.stoppingCriteria = stoppingCriteria
        self.__funcDict__ = dict()
        self.__gradDict__ = dict()

    def next_point(self, point, learningRate):
        if learningRate < 0:
            raise Exception("Learning rate must be positive")
        return np.add(point, np.multiply(self.vector, learningRate))

    # Возвращает номер итерации алгоритма
    def epoch(self):
        return len(self.__history__) - 1

    # При необходимости вычисляет функцию и увеличивает соответствующий счетчик
    def func(self, x_k):
        x_k = tuple(x_k)
        if x_k in self.__funcDict__:
            return self.__funcDict__[x_k]
        self.__funcCalculation__ += 1
        f = self.__funcFunc__(x_k)
        self.__funcDict__[x_k] = f
        return f

    # При необходимости вычисляет градиент и увеличивает соответствующий счетчик
    def grad(self, x_k):
        x_k = tuple(x_k)
        if x_k in self.__gradDict__:
            return self.__gradDict__[x_k]
        self.__gradCalculation__ += 1
        g = self.__gradFunc__(x_k)
        self.__gradDict__[x_k] = g
        return g

    def history(self):
        return self.__history__

    def func_calculations(self):
        return self.__funcCalculation__

    def grad_calculations(self):
        return self.__gradCalculation__

    # Запускает алгоритм, если iterations > 0, то к условию, содержащемуся в stoppingCriteria добавится условие на кол-во итераций алгоритма
    def __call__(self, startPoint, iterations):
        prev_stopping_criteria = self.stoppingCriteria
        if iterations > 0:
            self.stoppingCriteria = IterationsPlus(iterations, prev_stopping_criteria)
        point = startPoint
        self.__history__ = [startPoint] # История посещенных точек
        self.__funcCalculation__ = 0    # Счетчик вычислений функции
        self.__gradCalculation__ = 0    # Счётчик вычислений градиента
        while True:
            self.vector = np.multiply(self.grad(point), -1)  # Направление следующего движения
            point = self.next_point(point, self.learningRateCalculator.learning_rate(self))
            b = self.stoppingCriteria(self, point)
            self.__history__.append(point)
            if b:
                break
        self.stoppingCriteria = prev_stopping_criteria
        return self.__history__, self.__funcCalculation__, self.__gradCalculation__
