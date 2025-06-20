import numpy as np

from lab1.GradientDescent import GradientDescent, GDException

func_table = [
    [lambda x: x[0] ** 2 - 10,
     lambda x: 2 * x[0],
     None
     ],
    # lambdas = ((a+c) +- sqrt( (a-c)^2 + 4b^2 ))/2
    # x^2 + y^2; a = 1, b = 0, c = 1; lambdas = 1, 1; Число обусловленности = 1
    [lambda x: x[0] ** 2 + x[1] ** 2,
     lambda x: [2 * x[0], 2 * x[1]],
     lambda vector: [[2, 0], [0, 2]]
     ],
    # 3x^2 - 4xy + 10y^2; a = 3; b = -4; c = 10; lambdas = 2.3699, 23.63; Число обусловленности = 9,97
    [lambda x: 3 * x[0] ** 2 + 10 * x[1] ** 2 - 4 * x[0] * x[1],
     lambda x: [6 * x[0] - 4 * x[1], 20 * x[1] - 4 * x[0]],
     lambda vector: [[6, -4], [-4, 20]]],
    # (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    [lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
     lambda x: [2 * (x[0] ** 2 + x[1] - 11) * (2 * x[0]) + 2 * (x[0] + x[1] ** 2 - 7),
                2 * (x[0] + x[1] ** 2 - 7) * (2 * x[1]) + 2 * (x[0] ** 2 + x[1] - 11)],
     lambda x: [[12 * x[0] + 4 * (x[1] - 11) + 2, 4 * x[0] + 4 * x[1]],
                [4 * x[0] + 4 * x[1], 12 * x[1] + 4 * (x[0] - 7) + 2]]],
    # 20 + (x^2 − 10cos(2πx)) + (y^2 − 10cos(2πy))
    [lambda x: 20 + (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1])),
     lambda x: [2 * x[0] + 20 * np.pi * np.sin(2 * np.pi * x[0]),
                2 * x[1] + 20 * np.pi * np.sin(2 * np.pi * x[1])],
     lambda x: [[2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x[0]), 0],
                [0, 2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x[1])]]],
]

reals = [
        [(0,)],
        [(0, 0)],
        [(0, 0)],
        [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)],
        [(0, 0)]
    ]


def grad_des_instance(func_number, learning_rate, stopping_criteria):
    gd = GradientDescent(func_table[func_number][0], func_table[func_number][1], learning_rate, stopping_criteria)
    return gd


def run(gd, point, iterations):
    return gd(point, iterations)


def print_r(r, name=None):
    if name is not None:
        print(name)
    print("кол-во итераций + конечная точка")
    print(len(r[0]), r[0][-1])
    print("кол-во вызовов функции и ее производной")
    print(r[1:3])
    r = r[3:]
    if len(r):
        print("остальная информация")  # кол-во вычислений гессиана(если использовался)
        print(r)
    print("---------------------------------------")


def print_res(gd_inst, point, iterations, name=None):
    try:
        r = run(gd_inst, point, iterations)
        print_r(r, name)
    except GDException as _:
        print("NaN")
