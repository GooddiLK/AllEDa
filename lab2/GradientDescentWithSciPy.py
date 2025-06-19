from lab2.Utils2 import *
from scipy.optimize import minimize
import numpy as np


# принимает функцию и метод для scipy.optimize
# возвращает: точки, количество вычислений функции, градиента и гессиана(если использовался)
class GradientDescentWithSciPy:
    def __init__(self, func, method):
        self.__funcFunc__ = func
        self.method = method

    def __call__(self, startPoint, iterations):
        x, y = sp.symbols('x y')
        func_numeric = sp.lambdify((x, y), self.__funcFunc__, modules="numpy")
        grad_func_numeric = compute_gradient(self.__funcFunc__, (x, y))
        hess_func_numeric = compute_hessian(self.__funcFunc__, (x, y))
        # TODO: Версия newtoneMethodAdding:
        # hess_func_numeric = None
        # if self.method == "Newton-CG":
        #     hess_func_numeric = compute_hessian(self.__funcFunc__, (x, y))

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

        return history, result.nfev, result.njev, result.nhev
        # TODO: Версия newtoneMethodAdding:
        # if self.method == "Newton-CG":
        #     return history, result.nfev, result.njev, result.nhev
        # else:
        #     return history, result.nfev, result.njev, 0
