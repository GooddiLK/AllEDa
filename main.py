import math

import numpy as np
import sympy as sp

from matplotlib.colors import LinearSegmentedColormap
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot as plt

from lab1.GradientDescent import GradientDescent
from lab2.GradientDescent import GradientDescentWithScipy
from lab1.LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingGeom, \
    LearningRateSchedulingExponential, LearningRateSchedulingPolynomial
from lab1.OneDimensional import Armijo, Wolfe
from lab2.OneDimensional import BFGS
from lab1.StoppingCriteria import Iterations, SequenceEps, SequenceValueEps
from lab2.utils import *

x, y = sp.symbols('x y')

func_table = [
    [lambda x: x[0] ** 2 - 10, lambda x: 2 * x[0]],
    # lambdas = ((a+c) +- sqrt( (a-c)^2 + 4b^2 ))/2
    # x^2 + y^2; a = 1, b = 0, c = 1; lambdas = 1, 1; Число обусловленности = 1
    [lambda x: x[0] ** 2 + x[1] ** 2, lambda x: [2 * x[0], 2 * x[1]]],
    # 3x^2 - 4xy + 10y^2; a = 3; b = -4; c = 10; lambdas = 2.3699, 23.63; Число обусловленности = 9,97
    [lambda x: 3 * x[0] ** 2 + 10 * x[1] ** 2 - 4 * x[0] * x[1],
     lambda x: [6 * x[0] - 4 * x[1], 20 * x[1] - 4 * x[0]]],

    [lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
     lambda x: [2 * (x[0] ** 2 + x[1] - 11) * (2 * x[0]) + 2 * (x[0] + x[1] ** 2 - 7),
                2 * (x[0] + x[1] ** 2 - 7) * (2 * x[1]) + 2 * (x[0] ** 2 + x[1] - 11)]],
]

sympy_func = [
    x**2 + y**2,
    3 * x ** 2 + 10 * y ** 2 - 4 * x * y,
    (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
]

def print_res(gd_inst, point, iterations):
    r = gd_inst(point, iterations)
    print("кол-во итераций + конечная точка")
    print(len(r[0]), r[0][-1])
    print("кол-во вызовов функции и ее производной")
    print(r[1:])
    print("---------------------------------------")


def run(func_number, learning_rate, stopping_criteria, point, iterations):
    gd = GradientDescent(func_table[func_number][0], func_table[func_number][1], learning_rate, stopping_criteria)
    print_res(gd, point, iterations)

def to2(func):
    return lambda x, y: func([x, y])


def show(func, rng, grid, last_points, r):
    print(r)
    if last_points > 0:
        r = r[-last_points:]
    lx, ly = r[-1]
    xaxis = np.array(arange(-rng + lx, rng + lx, grid), dtype=np.float64)
    yaxis = np.array(arange(-rng + ly, rng + ly, grid), dtype=np.float64)
    x, y = np.array(meshgrid(xaxis, yaxis), dtype=np.float64)
    results = np.array(to2(func)(x, y), dtype=np.float64)
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(x, y, results, cmap='viridis', alpha=0.5)
    rx = np.array([i[0] for i in r], dtype=np.float64)
    ry = np.array([i[1] for i in r], dtype=np.float64)
    rz = np.array(to2(func)(rx, ry), dtype=np.float64)
    indices = np.linspace(0, 1, len(r))
    indices = np.array([math.exp(-i) for i in indices], dtype=np.float64)
    colors = ["green", "red"]
    cmap_custom = LinearSegmentedColormap.from_list("RedToGreen", colors)
    axis.scatter(
        rx, ry, rz,
        c=indices,
        cmap=cmap_custom,
    )
    plt.show()


if __name__ == "__main__":
    rng = 2

    f_num = 1
    test_point = [np.longdouble(10), np.longdouble(5)]
    iter_max = 10 ** 4
    #
    # run(f_num, LearningRateSchedulingConstant(0.025), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingGeom(1.1, 1/5), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingExponential(0.001), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingPolynomial(0.5, 2), SequenceValueEps(0.0001), test_point, iter_max)
    # a = 1
    # c1 = 0.001
    # c2 = 0.4
    # e = 0.001
    # run(1, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(2, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(3, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, Wolfe(1, 0.001, 0.1, 0.0001), SequenceValueEps(0.0001), test_point, iter_max)

    # Newton-CG с использованием scipy
    # gd_scipy = GradientDescentWithScipy(func=sympy_func[2] , method="Newton-CG")

    # BFGS с использованием scipy
    gd_scipy = GradientDescentWithScipy(func=sympy_func[2], method="BFGS")
    res = gd_scipy((1, 1), iter_max)
    show(func_table[3][0], rng, rng/20, 300, gd_scipy(startPoint=test_point, iterations=iter_max)[0])

    # gd = GradientDescent(func_table[3][0], func_table[3][1], Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -6))
    gd = GradientDescent(func_table[3][0], func_table[3][1], BFGS(sympy_func[2], 1/2, 0.6, 0.0001, 0.5), SequenceEps(10 ** -6))
    show(func_table[3][0], rng, rng / 20, 300, gd(startPoint=test_point, iterations=5000)[0])
