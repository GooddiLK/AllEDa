from GradientDescent import *
from OneDimensional import *
from StoppingCriteria import *

import numpy as np
# import sympy as sp

from utils import *
from Examples import func_table
from Pyplot import show

x, y = sp.symbols('x y')

sympy_func = [
    x**2 + y**2,
    3 * x ** 2 + 10 * y ** 2 - 4 * x * y,
    (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
]

if __name__ == "__main__":
    f_num = 1
    test_point = [np.longdouble(3), np.longdouble(-2)]
    iter_max = 10 ** 4
    #
    # run(f_num, LearningRateSchedulingConstant(0.025), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingGeom(1.1, 1/5), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingExponential(1, 0.001), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, LearningRateSchedulingPolynomial(0.5, 2), SequenceValueEps(0.0001), test_point, iter_max)
    # a = 1
    # c1 = 0.001
    # c2 = 0.4
    # e = 0.001
    # run(1, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(2, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(3, Wolfe(a, c1, c2, e), SequenceValueEps(0.0001), test_point, iter_max)
    # run(f_num, Wolfe(1, 0.001, 0.1, 0.0001), SequenceValueEps(0.0001), test_point, iter_max)

    gd = GradientDescent(func_table[2][0], func_table[2][1], Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -6))
    # graphic:
    rng = 2
    show(func_table[3][0], rng, rng/20, 300, gd([3, 3], 5 * 10**3)[0])

    # Newton-CG с использованием scipy
    # gd_scipy = GradientDescentWithScipy(func=sympy_func[2] , method="Newton-CG")

    # BFGS с использованием scipy
    # gd_scipy = GradientDescentWithScipy(func=sympy_func[2], method="BFGS")
    # res = gd_scipy((1, 1), iter_max)
    # show(func_table[3][0], rng, rng/20, 300, gd_scipy(startPoint=test_point, iterations=iter_max)[0])

    # gd = GradientDescent(func_table[3][0], func_table[3][1], Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -6))
    gd = GradientDescent(func_table[3][0], func_table[3][1], BFGS(sympy_func[2], 1/2, 0.6, 0.0001, 0.5), SequenceEps(10 ** -6))
    show(func_table[3][0], rng, rng/20, 300, gd(startPoint=test_point, iterations=5000)[0])

