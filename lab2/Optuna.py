import optuna as opt
from random import randint

from lab1.Examples import *
from lab1.GradientDescent import GDException
from lab1.LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingExponential
from lab1.OneDimensional import Armijo, Wolfe
from lab1.StoppingCriteria import *
from lab2.BFGS import BFGS
from lab2.main import sympy_func

max_iterations = 5 * 10 ** 3
eps = 10 ** -4
stopping_criteria = SequenceEps(eps)
big_constant = 10 ** 6


def optimizing_func(func_calc, grad_calc, found, real):
    return func_calc + grad_calc + big_constant * np.linalg.norm(real - found)


def run_study(objective, n_trials=50):
    study = opt.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Лучшие параметры:", study.best_params)
    print("Лучшее значение:", study.best_value)


# Runs objective with provided gradient descent and point it should find
def run_objective(gd, real, point_from=None, delta=50, number_of=40):
    if point_from is None:
        point_from = real
    dims = len(real)
    real = np.array(real).astype(np.longdouble)
    point_from = np.array(point_from).astype(np.longdouble)
    result = 0
    for i in range(number_of):
        point = [0] * dims
        for j in range(dims):
            point[j] = randint(int(point_from[j] - delta), int(point_from[j] + delta))
        point = np.array(point).astype(np.longdouble)
        try:
            h, func_calc, grad_calc = run(gd, point, max_iterations)
        except GDException as _:
            result += big_constant ** 2
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


# Optimizing GD with Armijo on function 2
def objective3(trial):
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    q = trial.suggest_float('q', 1e-5, 1, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=True)
    armijo = Armijo(alpha_0, q, epsilon, c1)
    gd = grad_des_instance(2, armijo, stopping_criteria)
    return run_objective(gd, [0, 0])


# Optimizing GD with Wolfe on function 3
def objective4(trial):
    eps = 10 ** -5
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c2 = trial.suggest_float('c2', 1e-6, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-8, c2, log=True)
    wolfe = Wolfe(alpha_0, c1, c2, epsilon)
    gd = grad_des_instance(3, wolfe, stopping_criteria)
    return run_objective(gd, [3, 2], delta=0, point_from=[3, 4], number_of=1)


# Optimizing BFGS with Wolfe on function 3
def objective5(trial):
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=True)
    q = trial.suggest_float('q', 1e-5, 1, log=True)
    func_number = 3
    func = sympy_func[func_number]
    bfgs = BFGS(func, alpha_0, q, epsilon, c1)
    gd = grad_des_instance(func_number, bfgs, stopping_criteria)
    return run_objective(gd, [3, 2], point_from=[3, 4], number_of=1, delta=0)