# В этом файле содержатся алгоритмы планирования шага
# GradientDescent требует, чтобы объект планирования шага обладал методом learning_rate,
# принимающим объект типа GradientDescent и возвращающим текущий шаг

from math import sqrt, exp


# Константный шаг
# Размер шага должен быть порядка сотых
class LearningRateSchedulingConstant:
    def __init__(self, learning_rate):
        self.learning_rate_c = learning_rate

    def learning_rate(self, gd):
        return self.learning_rate_c


# Пример кусочно-постоянного шага: геометрическая прогрессия
class LearningRateSchedulingGeom:
    def __init__(self, k, learning_rate_0):
        self.k = k
        self.learning_rate_0 = learning_rate_0

    def learning_rate(self, gd):
        return self.learning_rate_0 / (self.k ** (gd.epoch() - 1))


# Планирование шага - экспоненциальное затухание
# Экспонента должна быть маленькой, порядка десятых
class LearningRateSchedulingExponential:
    def __init__(self, h0, lambdaParameter):
        self.h0 = h0
        self.lambdaParameter = lambdaParameter

    def learning_rate(self, gd):
        epoch = gd.epoch()
        return self.h0 * exp(-self.lambdaParameter * epoch)


# Вспомогательная функция для функционального планирования шага
def h0(epoch):
    return 1 / sqrt(epoch + 1)


# Планирование шага - полиномиальное затухание
class LearningRateSchedulingPolynomial:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def learning_rate(self, gd):
        epoch = gd.epoch()
        return h0(epoch) * (self.beta * epoch + 1) ** (-self.alpha)
