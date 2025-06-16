import numpy as np
from lab3.SGD import StochasticGradientDescent
from lab1.LearningRateScheduling import LearningRateSchedulingConstant
from lab1.StoppingCriteria import Iterations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Генерация данных
X, y = make_regression(n_samples=5000, n_features=100, noise=0.1)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.astype(float)

# Параметры SGD
sgd = StochasticGradientDescent(
    func=lambda w: np.mean((np.dot(X_scaled, w) - y) ** 2),  # MSE
    grad=lambda w: (2 / len(X_scaled)) * np.dot(X_scaled.T, (np.dot(X_scaled, w) - y)),
    learning_rate_calculator=LearningRateSchedulingConstant(0.01),
    stopping_criteria=Iterations(100)
)

batch_sizes = [1, 10, 50, 100]
results = []
for bs in batch_sizes:
    res = sgd(
        start_weight=np.zeros(X_scaled.shape[1]),
        X=X_scaled,
        y=y,
        batch_size=bs,
        iterations=100
    )
    results.append({
        'batch_size': bs,
        'mse': res['mse'],
        'time_sec': res['time_sec'],
        'memory_mb': res['memory_mb'],
        'operations': res['total_operations']
    })

df_results = pd.DataFrame(results)
print(df_results)