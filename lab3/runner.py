import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import tracemalloc
from memory_profiler import memory_usage
from functools import wraps

from lab3.SGD import StochasticGradientDescent
from lab3.MomentumSGD import MomentumSGD

def monitor_resources(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        baseline_memory = memory_usage(-1, interval=0.1, timeout=1)[0]
        start_time = time.time()

        result = func(*args, **kwargs)

        time_sec = time.time() - start_time
        mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if isinstance(result, dict):
            result.update({
                'time_sec': time_sec,
                'memory_mb': mem_after - baseline_memory,
                'allocated_memory_mb': peak / 1024 / 1024
            })
        return result
    return wrapper


@monitor_resources
def run_torch_optimizer(optimizer_class, X, y, batch_size, epochs=100):
    model = nn.Linear(X.shape[1], 1)
    criterion = nn.MSELoss()

    if optimizer_class == optim.Adam:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_class == optim.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        optimizer = optimizer_class(model.parameters())

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

    with torch.no_grad():
        y_pred = model(X)
        mse = criterion(y_pred, y).item()

    return {
        'type': optimizer.__class__.__name__,
        'batch_size': batch_size,
        'mse': mse,
        'operations': 'N/A'
    }

@monitor_resources
def run_momentum_sgd(X, y, batch_size, learning_rate, stopping_criteria, regularization=None, momentum=0.9, epochs=10000):
    sgd = MomentumSGD(
        learning_rate_calculator=learning_rate,
        stopping_criteria=stopping_criteria,
        regularization=regularization,
        momentum=momentum,
    )

    start_weight = np.random.randn(X.shape[1]) * 0.01

    res = sgd(
        start_weight=start_weight,
        X=X,
        y=y.flatten(),
        batch_size=batch_size,
        iterations=epochs
    )

    return {
        'type': 'MomentumSGD',
        'batch_size': batch_size,
        'mse': res['mse'],
        'operations': res.get('total_operations', 0)
    }

@monitor_resources
def run_custom_sgd(X, y, batch_size, learning_rate, stopping_criteria, regularization=None, epochs=10000):
    sgd = StochasticGradientDescent(
        learning_rate_calculator=learning_rate,
        stopping_criteria=stopping_criteria,
        regularization=regularization,
    )

    start_weight = np.random.randn(X.shape[1]) * 0.01
    res = sgd(
        start_weight=start_weight,
        X=X,
        y=y.flatten(),
        batch_size=batch_size,
        iterations=epochs
    )

    return {
        'type': 'Custom SGD',
        'batch_size': batch_size,
        'mse': res['mse'],
        'operations': res.get('total_operations', 0)
    }