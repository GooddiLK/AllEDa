import numpy as np

from lab1.StoppingCriteria import IterationsPlus

class StochasticGradientDescent:
    def __init__(self, learning_rate_calculator, stopping_criteria, regularization=None, reg_param=0.001):
        self.learningRateCalculator = learning_rate_calculator
        self.stoppingCriteria = stopping_criteria
        self.regularization = regularization
        self.reg_param = reg_param

    def _apply_regularization(self, weights):
        if self.regularization == 'l1':
            return self.reg_param * np.sign(weights)
        elif self.regularization == 'l2':
            return self.reg_param * weights
        elif self.regularization == 'elastic':
            l1 = self.reg_param * np.sign(weights)
            l2 = self.reg_param * weights
            return 0.5 * (l1 + l2)
        return 0

    def next_point(self, point, gradient, learning_rate):
        if learning_rate < 0:
            raise Exception("Learning rate must be positive")
        return np.add(point, np.multiply(gradient, -learning_rate))

    def epoch(self):
        return len(self.__history__) - 1

    def history(self):
        return self.__history__

    def __call__(self, start_weight, X, y, batch_size, iterations):
        operation_count = 0

        prev_stopping_criteria = self.stoppingCriteria
        if iterations > 0:
            self.stoppingCriteria = IterationsPlus(iterations, prev_stopping_criteria)

        weight = start_weight
        self.__history__ = [start_weight]
        n_samples = len(X)
        n_features = X.shape[1] # количество признаков

        while True:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Предсказание: матричное умножение (batch_size * n_features)
                predictions = np.dot(X_batch, weight)

                # Ошибка: вычитание (batch_size)
                errors = predictions - y_batch

                # Градиент: матричное умножение (batch_size * n_features)
                gradient = (2 / len(y_batch)) * np.dot(X_batch.T, errors)

                # Регуляризация (n_features)
                gradient += self._apply_regularization(weight)

                # Обновление весов (n_features)
                learningRate = self.learningRateCalculator.learning_rate(self)
                weight = self.next_point(weight, gradient, learningRate)

                operation_count += (batch_size * (2 * n_features + 1)
                                    + 2 * n_features)

            self.__history__.append(weight)
            if self.stoppingCriteria(self, weight) or np.mean((np.dot(X, weight) - y)**2) < 0.0001:
                break

        # Вычисление MSE на всех данных
        y_pred = np.dot(X, weight)
        mse = np.mean((y_pred - y) ** 2)

        self.stoppingCriteria = prev_stopping_criteria

        return {
            "history": self.__history__,
            "final_weights": weight,
            "mse": mse,
            "total_operations": operation_count,
        }