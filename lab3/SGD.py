import numpy as np
import time
from memory_profiler import memory_usage

from lab1.StoppingCriteria import IterationsPlus


class StochasticGradientDescent:
    def __init__(self, func, grad, learning_rate_calculator, stopping_criteria, regularization=None, reg_param=0.01):
        self.learningRateCalculator = learning_rate_calculator
        self.stoppingCriteria = stopping_criteria
        self.regularization = regularization
        self.reg_param = reg_param
        self.__funcDict__ = dict()
        self.__gradDict__ = dict()

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
        start_time = time.time()
        mem_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        operation_count = 0

        prev_stopping_criteria = self.stoppingCriteria
        if iterations > 0:
            self.stoppingCriteria = IterationsPlus(iterations, prev_stopping_criteria)

        weight = start_weight
        self.__history__ = [start_weight]
        n_samples = len(X)
        n_features = X.shape[1]

        while True:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Предсказание и ошибка (2 * batch_size * n_features операций)
                predictions = np.dot(X_batch, weight)
                errors = predictions - y_batch
                operation_count += 2 * batch_size * n_features

                # Градиент (batch_size * n_features операций)
                gradient = (2 / len(y_batch)) * np.dot(X_batch.T, errors)
                operation_count += batch_size * n_features

                gradient += self._apply_regularization(weight)

                learningRate = self.learningRateCalculator.learning_rate(self)
                weight = self.next_point(weight, gradient, learningRate)

            self.__history__.append(weight)
            if self.stoppingCriteria(self, weight):
                break

        total_time = time.time() - start_time
        mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        mem_used = mem_after - mem_before  # MB

        # Вычисление MSE на всех данных
        y_pred = np.dot(X, weight)
        mse = np.mean((y_pred - y) ** 2)

        self.stoppingCriteria = prev_stopping_criteria

        return {
            "history": self.__history__,
            "final_weights": weight,
            "mse": mse,
            "time_sec": total_time,
            "memory_mb": mem_used,
            "total_operations": operation_count,
        }