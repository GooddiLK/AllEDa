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
big_constant = 10 ** 18


def optimizing_func(func_calc, grad_calc, found, real):
    return func_calc + grad_calc + big_constant * np.linalg.norm(real - found)


def run_study(pre_objective, func_number, real, n_trials=50):
    study = opt.create_study(direction="minimize")
    def objective(trial):
        return pre_objective(trial, func_number, real)
    study.optimize(objective, n_trials=n_trials)
    return study


def print_study(study):
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


def trial_learning_rate_scheduling_constant(trial):
    learning_rate_0 = trial.suggest_float('learning_rate', 1e-10, 1, log=True)
    return LearningRateSchedulingConstant(learning_rate_0)


# Optimizing constant learning rate
def objective_learning_rate_scheduling_constant(trial, func_number, real):
    learning_rate = trial_learning_rate_scheduling_constant(trial)
    gd = grad_des_instance(func_number, learning_rate, stopping_criteria)
    return run_objective(gd, real)


def learn_learning_rate_scheduling_constant(study, func_number, stop):
    learning_rate = LearningRateSchedulingConstant(study.best_params['learning_rate'])
    return grad_des_instance(func_number, learning_rate, stop)


def trial_learning_rate_scheduling_exponential(trial):
    h0 = trial.suggest_float('h0', 1e-10, 1, log=True)
    lambda_param = trial.suggest_float('lambda_param', 1e-10, 1, log=True)
    return LearningRateSchedulingExponential(h0, lambda_param)


# Optimizing exponential learning rate
def objective_learning_rate_scheduling_exponential(trial, func_number, real):
    learning_rate = trial_learning_rate_scheduling_exponential(trial)
    gd = grad_des_instance(func_number, learning_rate, stopping_criteria)
    return run_objective(gd, real)


def learn_learning_rate_scheduling_exponential(study, func_number, stop):
    learning_rate = LearningRateSchedulingExponential(study.best_params['h0'], study.best_params['lambda_param'])
    return grad_des_instance(func_number, learning_rate, stop)


def trial_armijo(trial):
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    q = trial.suggest_float('q', 1e-5, 1, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=True)
    return Armijo(alpha_0, q, epsilon, c1)

# Optimizing GD with Armijo
def objective_armijo(trial, func_number, real):
    armijo = trial_armijo(trial)
    gd = grad_des_instance(func_number, armijo, stopping_criteria)
    return run_objective(gd, real)


def learn_armijo(study, func_number, stop):
    armijo = Armijo(study.best_params["alpha_0"], study.best_params["q"], study.best_params["epsilon"], study.best_params["c1"])
    return grad_des_instance(func_number, armijo, stop)


def trial_wolfe(trial):
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c2 = trial.suggest_float('c2', 1e-6, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-8, c2, log=True)
    return Wolfe(alpha_0, c1, c2, epsilon)


# Optimizing GD with Wolfe
def objective_wolfe(trial, func_number, real):
    wolfe = trial_wolfe(trial)
    gd = grad_des_instance(func_number, wolfe, stopping_criteria)
    return run_objective(gd, real)


def learn_wolfe(study, func_number, stop):
    wolfe = Wolfe(study.best_params["alpha_0"], study.best_params["c1"], study.best_params["c2"], study.best_params["epsilon"])
    return grad_des_instance(func_number, wolfe, stop)


def trial_bfgs(trial, func_number):
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=True)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=True)
    q = trial.suggest_float('q', 1e-5, 1, log=True)
    return BFGS(sympy_func[func_number], alpha_0, q, epsilon, c1)

# Optimizing BFGS
def objective_bfgs(trial, func_number, real):
    bfgs = trial_bfgs(trial, func_number)
    gd = grad_des_instance(func_number, bfgs, stopping_criteria)
    return run_objective(gd, real)


def learn_bfgs(study, func_number, stop):
    bfgs = BFGS(sympy_func[func_number], study.best_params["alpha_0"], study.best_params["q"], study.best_params["epsilon"], study.best_params["c1"])
    return grad_des_instance(func_number, bfgs, stop)

# TODO: Add newton
