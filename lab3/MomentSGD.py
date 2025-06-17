import numpy as np
from lab3.SGD import StochasticGradientDescent

class MomentumSGD(StochasticGradientDescent):

    def __init__(self, learning_rate_calculator, stopping_criteria,
                 momentum=0.9, **kwargs):
        """
            momentum: коэффициент сохранения импульса (0 ; 1)
            nesterov: использовать ли ускорение Нестерова
        """
        super().__init__(learning_rate_calculator, stopping_criteria, **kwargs)
        self.momentum = momentum
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





