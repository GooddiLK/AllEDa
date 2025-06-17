import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from memory_profiler import memory_usage

# Генерация данных
t0 = time.time()
X, y = make_regression(n_samples=100000, n_features=20)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.astype(np.float32).reshape(-1, 1)
t1 = time.time()
print(f"Time gen: {t1 - t0} seconds")

# Конвертация в тензоры PyTorch
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y)

from lab3.SGD import StochasticGradientDescent
from lab3.MomentSGD import MomentumSGD
from lab1.LearningRateScheduling import LearningRateSchedulingConstant
from lab1.StoppingCriteria import Iterations


def run_torch_optimizer(optimizer_class, X, y, batch_size, epochs=100):
    # Простая линейная модель
    model = nn.Linear(X.shape[1], 1)
    criterion = nn.MSELoss()

    if optimizer_class == optim.Adam:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_class == optim.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        optimizer = optimizer_class(model.parameters())

    start_time = time.time()
    mem_before = memory_usage(-1, interval=0.1, timeout=1)[0]

    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X[indices], y[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]
    time_sec = time.time() - start_time

    with torch.no_grad():
        y_pred = model(X_tensor)
        mse = criterion(y_pred, y_tensor).item()

    return {
        'type': optimizer.__class__.__name__,
        'batch_size': batch_size,
        'mse': mse,
        'time_sec': time_sec,
        'memory_mb': mem_after - mem_before,
        'operations': 'N/A'
    }

def run_momentum_sgd(X, y, batch_size, momentum=0.9,  iterations=100):
    sgd = MomentumSGD(
        learning_rate_calculator=LearningRateSchedulingConstant(0.01),
        stopping_criteria=Iterations(iterations),
        momentum=momentum,
    )

    start_time = time.time()
    mem_before = memory_usage(-1, interval=0.1, timeout=1)[0]

    res = sgd(
        start_weight=np.zeros(X.shape[1]),
        X=X,
        y=y.flatten(),
        batch_size=batch_size,
        iterations=iterations
    )

    time_sec = time.time() - start_time
    mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]

    return {
        'type': 'MomentumSGD' ,
        'batch_size': batch_size,
        'mse': res['mse'],
        'time_sec': time_sec,
        'memory_mb': mem_after - mem_before,
        'operations': res.get('total_operations', 0)
    }

def run_custom_sgd(X, y, batch_size, regularization=None, iterations=100):
    sgd = StochasticGradientDescent(
        learning_rate_calculator=LearningRateSchedulingConstant(0.01),
        stopping_criteria=Iterations(iterations),
        regularization=regularization,
    )
    start_time = time.time()
    mem_before = memory_usage(-1, interval=0.1, timeout=1)[0]
    res = sgd(
        start_weight=np.zeros(X.shape[1]),
        X=X,
        y=y.flatten(),
        batch_size=batch_size,
        iterations=iterations
    )
    time_sec = time.time() - start_time
    mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]
    mem_used = mem_after - mem_before
    return {
        'type': 'Custom SGD',
        'batch_size': batch_size,
        'mse': res['mse'],
        'time_sec': time_sec,
        'memory_mb': mem_used,
        'operations': res.get('total_operations', 0)
    }


batch_sizes = [100, 1000, 5000, 10000]
results = []

for bs in batch_sizes:
    results.append(run_custom_sgd(X_scaled, y, bs))
    results.append(run_momentum_sgd(X_tensor, y, bs, 0.9))

    # PyTorch оптимизаторы
    results.append(run_torch_optimizer(optim.SGD, X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(lambda p: optim.SGD(p, momentum=0.9, lr=0.001), X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(lambda p: optim.SGD(p, momentum=0.9, lr=0.001, nesterov=True), X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(optim.RMSprop, X_tensor, y_tensor, bs))
    results.append(run_torch_optimizer(optim.Adam, X_tensor, y_tensor, bs))

df_results = pd.DataFrame(results)
print(df_results[['type', 'batch_size', 'mse', 'time_sec', 'memory_mb']])
