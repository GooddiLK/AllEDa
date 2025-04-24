import optuna as opt
from random import randint

from lab1.Examples import *
from lab1.GradientDescent import GDException
from lab1.LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingExponential
from lab1.StoppingCriteria import *

max_iterations = 5 * 10 ** 3
eps = 10 ** -4
stopping_criteria = SequenceEps(eps)
very_big_constant = 1_000_000_000_000_000_000


def optimizing_func(func_calc, grad_calc, found, real):
    return func_calc + grad_calc + very_big_constant * np.linalg.norm(real - found)


def run_study(objective, n_trials=50):
    study = opt.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Лучшие параметры:", study.best_params)


# Runs objective with provided gradient descent and point it should find
def run_objective(gd, real, delta=50, number_of=40):
    dims = len(real)
    real = np.array(real).astype(np.longdouble)
    result = 0
    for i in range(number_of):
        point = [0] * dims
        for j in range(dims):
            point[j] = randint(int(real[j] - delta), int(real[j] + delta))
        point = np.array(point).astype(np.longdouble)
        try:
            h, func_calc, grad_calc = run(gd, point, max_iterations)
        except GDException as _:
            result += very_big_constant ** 2
            continue
        one_result = optimizing_func(func_calc, grad_calc, h[-1], real)
        result += one_result
    return result


# Optimizing constant learning rate on function 1
def objective1(trial):
    learning_rate_0 = trial.suggest_float('learning_rate', 1e-10, 1, log=True)
    learning_rate = LearningRateSchedulingConstant(learning_rate_0)
    gd = grad_des_instance(1, learning_rate, stopping_criteria)
    return run_objective(gd, [0, 0])


# Optimizing exponential learning rate on function 2
def objective2(trial):
    h0 = trial.suggest_float('h0', 1e-10, 1, log=True)
    lambda_param = trial.suggest_float('lambda_param', 1e-10, 1, log=True)
    learning_rate = LearningRateSchedulingExponential(h0, lambda_param)
    gd = grad_des_instance(2, learning_rate, stopping_criteria)
    return run_objective(gd, [0, 0])

