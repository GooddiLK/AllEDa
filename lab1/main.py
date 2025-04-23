import math

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from GradientDescent import GradientDescent
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingGeom, \
    LearningRateSchedulingExponential, LearningRateSchedulingPolynomial
from OneDimensional import Armijo, Wolfe
from StoppingCriteria import Iterations, SequenceEps, SequenceValueEps

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
    if last_points > 0:
        r = r[-last_points:]
    lx, ly = r[-1]
    xaxis = arange(-rng + lx, rng + lx, grid)
    yaxis = arange(-rng + ly, rng + ly, grid)
    x, y = meshgrid(xaxis, yaxis)
    results = to2(func)(x, y)
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(x, y, results, cmap='viridis', alpha=0.5)
    rx = np.array([i[0] for i in r])
    ry = np.array([i[1] for i in r])
    rz = to2(func)(rx, ry)
    indices = np.linspace(0, 1, len(r))
    indices = np.array([math.exp(-i) for i in indices])
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
    test_point = [np.longdouble(10), np.longdouble(10)]
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

    #
    gd = GradientDescent(func_table[2][0], func_table[2][1], Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -6))
    show(func_table[3][0], rng, rng/20, 300, gd([3, 3], 5 * 10**3)[0])
