import numpy as np
import optuna as opt
import math
from random import randint

from lab1.Examples import *
from lab1.GradientDescent import GDException
from lab1.LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingExponential
from lab1.OneDimensional import Armijo, Wolfe
from lab1.StoppingCriteria import *
from lab2.BFGS import BFGS
from lab2.main import sympy_func
from lab2.NewtoneMethod import newtoneMethodStart
import logging

max_iterations = 5 * 10 ** 3
eps = 10 ** -4
stopping_criteria = SequenceEps(eps)
big_constant = 10 ** 18
log = True
show_progress = True

# Функция по которой optuna оптимизирует параметры
def optimizing_func(func_calc, grad_calc, found, real):
    return func_calc + grad_calc + big_constant * np.linalg.norm(real - found)

# Запуск оптимизации optuna
def run_study(pre_objective, func_number, real, n_trials=50):
    logging.getLogger("optuna").setLevel(logging.WARNING)
    study = opt.create_study(direction="minimize")
    def objective(trial):
        return pre_objective(trial, func_number, real)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study

def run_study_newton(pre_objective, func_number, real, point_from0, point_from1, delta, n_trials=50):
    logging.getLogger("optuna").setLevel(logging.WARNING)
    study = opt.create_study(direction="minimize")
    def objective(trial):
        return pre_objective(trial, func_number, real, point_from0, point_from1, delta)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
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


# Все функции trial_* возвращает объект инициализируют trial optuna и возвращают объект градиентного спуска,
# который будет использоваться.
#
# Все функции objective_* возвращает объект, который optuna должна минимизировать.
#
# Все функции learn_* возвращает объект градиентного спуска, который optuna получила при оптимизации

def trial_learning_rate_scheduling_constant(trial):
    learning_rate_0 = trial.suggest_float('learning_rate', 1e-10, 1, log=log)
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
    h0 = trial.suggest_float('h0', 1e-10, 1, log=log)
    lambda_param = trial.suggest_float('lambda_param', 1e-10, 1, log=log)
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
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=log)
    q = trial.suggest_float('q', 1e-5, 1, log=log)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=log)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=log)
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
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=log)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=log)
    c2 = trial.suggest_float('c2', 1e-6, 1, log=log)
    c1 = trial.suggest_float('c1', 1e-8, c2, log=log)
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
    alpha_0 = trial.suggest_float('alpha_0', 1e-5, 10, log=log)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1, log=log)
    c1 = trial.suggest_float('c1', 1e-5, 1, log=log)
    q = trial.suggest_float('q', 1e-5, 1, log=log)
    return BFGS(sympy_func[func_number], alpha_0, q, epsilon, c1)

# Optimizing BFGS
def objective_bfgs(trial, func_number, real):
    bfgs = trial_bfgs(trial, func_number)
    gd = grad_des_instance(func_number, bfgs, stopping_criteria)
    return run_objective(gd, real)


def learn_bfgs(study, func_number, stop):
    bfgs = BFGS(sympy_func[func_number], study.best_params["alpha_0"], study.best_params["q"], study.best_params["epsilon"], study.best_params["c1"])
    return grad_des_instance(func_number, bfgs, stop)

def optimizing_func_newton(func_calc, grad_calc, hess_calc, found, real):
    return big_constant * np.linalg.norm(real - found) + func_calc + grad_calc + hess_calc

def run_objective_newton(newton, real, point_from0, point_from1, delta=50, number_of=40):
    dims = len(real)
    real = np.array(real).astype(np.longdouble)
    point_from0 = np.array(point_from0).astype(np.longdouble)
    point_from1 = np.array(point_from1).astype(np.longdouble)
    result = 0
    for i in range(number_of):
        point1 = [0] * dims
        for j in range(dims):
            point1[j] = randint(int(point_from1[j] - delta), int(point_from1[j] + delta))
        point1 = np.array(point1).astype(np.longdouble)
        try:
            h, func_calc, grad_calc, hess_calc = newton(point_from0,  point1)
        except GDException as _:
            result += big_constant ** 2
            continue
        one_result = optimizing_func_newton(func_calc, grad_calc, hess_calc, h[-1], real)
        result += one_result
    return result


def extract_newton(func_number, alpha_0, iteration_stop_limit, c2, c1, delta, learning_rate, trust_upper_bound,
    trust_lower_bound, trust_no_trust_bound, trust_changing_multiply_value):
    def newton(x0, x1):
        return newtoneMethodStart(
            func_table[func_number][0],
            func_table[func_number][1],
            func_table[func_number][2],
            x0,
            x1,
            alpha_0,
            c1,
            c2,
            delta,
            iteration_stop_limit,
            max_iterations,
            learning_rate,
            trust_upper_bound,
            trust_lower_bound,
            trust_no_trust_bound,
            trust_changing_multiply_value,
        )
    return newton

def trial_newton(trial, func_number):
    alpha_0 = trial.suggest_float('alpha_0', 1e-2, 10, log=log)
    iteration_stop_limit = trial.suggest_float('iteration_stop_limit', 1e-10, 0.1, log=log)
    c2 = trial.suggest_float('c2', 1e-2, 0.8, log=log)
    c1 = trial.suggest_float('c1', 1e-7, c2 / 10, log=log)
    delta = trial.suggest_float('delta', 1e-2, 10, log=log)
    learning_rate = trial.suggest_float('learning_rate', 1e-2, 5, log=log)
    trust_upper_bound = trial.suggest_float('trust_upper_bound', 0.5, 0.9999, log=log)
    trust_lower_bound = trial.suggest_float('trust_lower_bound', 1e-2, 0.5, log=log)
    trust_no_trust_bound = trial.suggest_float('trust_no_trust_bound', 1e-3, trust_lower_bound / 2, log=log)
    trust_changing_multiply_value = trial.suggest_float('trust_changing_multiply_value', 1, 10, log=log)
    # alpha_0 = 10
    # iteration_stop_limit = 0.000001
    # c2 = 0.7
    # c1 = 0.2
    # delta = 1
    # learning_rate = 0.1
    # trust_upper_bound = 0.75
    # trust_lower_bound = 1/4
    # trust_no_trust_bound = 1/16
    # trust_changing_multiply_value = 2

    return extract_newton(func_number, alpha_0, iteration_stop_limit, c2, c1, delta, learning_rate,
                          trust_upper_bound, trust_lower_bound, trust_no_trust_bound, trust_changing_multiply_value)

# Optimizing Newton
def objective_newton(trial, func_number, real, point_from0, point_from1, delta):
    newton = trial_newton(trial, func_number)
    return run_objective_newton(newton, real, point_from0, point_from1, delta, 2)

def learn_newton(study, func_number, _):
    return extract_newton(
        func_number,
        study.best_params["alpha_0"],
        study.best_params["iteration_stop_limit"],
        study.best_params["c2"],
        study.best_params["c1"],
        study.best_params["delta"],
        study.best_params["learning_rate"],
        study.best_params["trust_upper_bound"],
        study.best_params["trust_lower_bound"],
        study.best_params["trust_no_trust_bound"],
        study.best_params["trust_changing_multiply_value"],
    )