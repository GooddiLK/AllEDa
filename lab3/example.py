import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd

from lab3.runner import (
    run_momentum_sgd,
    run_custom_sgd,
    run_torch_optimizer
)

from lab1.LearningRateScheduling import (
    LearningRateSchedulingConstant,
    LearningRateSchedulingExponential,
    LearningRateSchedulingPolynomial,
    LearningRateSchedulingGeom
)

import lab1.StoppingCriteria as StoppingCriteria

# X — данные на основе которых модель обучается
# y — цель обучения

X, y = make_regression(n_samples=10000, n_features=20)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.astype(np.float32).reshape(-1, 1)

# Конвертация в тензоры PyTorch
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y)

batch_sizes = [100, 256, 512, 1000]
results = []

epochs = 10000
sc = StoppingCriteria.Iterations(epochs)
lr =  LearningRateSchedulingConstant(0.0001)
reg = None # l1, l2, elastic

for bs in batch_sizes:
    results.append(run_custom_sgd(X_scaled, y, bs, lr, sc, reg))
    results.append(run_momentum_sgd(X_tensor, y, bs, lr, sc, reg))

    # PyTorch оптимизаторы
    results.append(run_torch_optimizer(optim.SGD, X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(lambda p: optim.SGD(p, momentum=0.9, lr=0.001), X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(lambda p: optim.SGD(p, momentum=0.9, lr=0.001, nesterov=True), X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(optim.RMSprop, X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(optim.Adam, X_tensor, y_tensor, bs))

df_results = pd.DataFrame(results)
# mse - точность модели
pd.set_option('display.max_columns', None)
print(df_results[['type', 'batch_size', 'mse', 'time_sec', 'memory_mb', 'allocated_memory_mb']])
