import numpy as np
from lab3.SGD import StochasticGradientDescent


class MomentumSGD(StochasticGradientDescent):
    def __init__(self, learning_rate_calculator, stopping_criteria, regularization=None, reg_param=0.0001,
                 momentum=0.9):
        super().__init__(learning_rate_calculator, stopping_criteria, regularization, reg_param)
        self.momentum = momentum  # Коэффициент сохранения импульса (0 ; 1)
        self.velocity = None  # Вектор скорости

    def _init_velocity(self, n_features):
        self.velocity = np.zeros(n_features)

    def next_point(self, point, gradient, learning_rate):
        if self.velocity is None:
            self._init_velocity(len(point))

        # Обновление скорости
        self.velocity = self.momentum * self.velocity - learning_rate * gradient

        # Обновление параметров
        point += self.velocity

        return point
