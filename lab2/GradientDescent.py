# В этом файле содержится непосредственно алгоритм градиентного спуска

import numpy as np
from lab2.utils import *
from scipy.optimize import minimize
import sympy as sp


# принимает функцию и  метод для scipy.optimize
# возвращает: точки, количество вычеслений функции, градиента и гессиана(если использовался)
class GradientDescentWithScipy:
    def __init__(self, func, method):
        self.__funcFunc__ = func
        self.method = method

    def __call__(self, startPoint, iterations):
        x, y = sp.symbols('x y')
        func_numeric = sp.lambdify((x, y), self.__funcFunc__, modules="numpy")
        grad_func_numeric = compute_gradient(self.__funcFunc__, (x, y))
        hess_func_numeric = None
        if self.method == "Newton-CG":
            hess_func_numeric = compute_hessian(self.__funcFunc__, (x, y))
        else:
            hess_func_numeric = None

        def hess_f(x_arr):
            return np.array(hess_func_numeric(x_arr[0], x_arr[1]))

        def grad_f(x_arr):
            return np.array(grad_func_numeric(x_arr[0], x_arr[1]))
        def f(x_arr):
            return np.array(func_numeric(x_arr[0], x_arr[1]))

        history = []

        def callback(xk):
            history.append(np.copy(xk))

        result = minimize(
            fun=f,
            x0=np.array(startPoint, dtype=float),
            jac=grad_f,
            hess=hess_f,
            method=self.method,
            options={"maxiter": iterations},
            callback=callback,
        )

        if self.method == "Newton-CG":
            return history, result.nfev, result.njev, result.nhev
        else:
            return history, result.nfev, result.njev, 0