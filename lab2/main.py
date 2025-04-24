from lab2.Optuna import *
import sympy as sp
from lab1.Pyplot import *
from lab2.BFGS import BFGS

x, y = sp.symbols('x y')

sympy_func = [
    x**2 + y**2,
    3 * x ** 2 + 10 * y ** 2 - 4 * x * y,
    (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
]

if __name__ == "__main__":
    # run_study(objective1)
    run_study(objective2)

    test_point = [np.longdouble(3), np.longdouble(-2)]
    iter_max = 10 ** 4
    # Newton-CG с использованием scipy
    # gd_scipy = GradientDescentWithScipy(func=sympy_func[2] , method="Newton-CG")

    # BFGS с использованием scipy
    # gd_scipy = GradientDescentWithScipy(func=sympy_func[2], method="BFGS")
    # res = gd_scipy((1, 1), iter_max)
    # show(func_table[3][0], rng, rng/20, 300, gd_scipy(startPoint=test_point, iterations=iter_max)[0])

    # gd = GradientDescent(func_table[3][0], func_table[3][1], Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -6))
    gd = GradientDescent(func_table[3][0], func_table[3][1], BFGS(sympy_func[2], 1 / 2, 0.6, 0.0001, 0.5),
                         SequenceEps(10 ** -6))
    rng = 2
    show(func_table[3][0], rng, rng / 20, 300, gd(startPoint=test_point, iterations=5000)[0])
