from lab2.Optuna import *
from lab2.GradientDescentWithSciPy import *
import sympy as sp

x, y = sp.symbols('x y')

sympy_func = [
    None,
    x ** 2 + y ** 2,
    3 * x ** 2 + 10 * y ** 2 - 4 * x * y,
    (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
    20 + (x ** 2 - 10 * sp.cos(2 * sp.pi * x)) + (y ** 2 - 10 * sp.cos(2 * sp.pi * y)),
]

if __name__ == "__main__":
    reals = [(0, 0), (0, 0), (0, 0), (3, 2), (0, 0)]  # real mins
    points = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]  # points to call on
    stop = SequenceEps(10 ** -6)
    from Optuna import max_iterations

    for i in range(1, 5):
        def run_too_many_word(learn, objective):
            study = run_study(objective, i, reals[i])
            print_study(study)
            return learn(study, i, stop)


        print(i)
        print("------------------------------------------------------------------------")
        # gd = run_too_many_word(learn_learning_rate_scheduling_constant, objective_learning_rate_scheduling_constant)
        # print_res(gd, points[i], max_iterations, "learning_rate_scheduling_constant")
        # gd = run_too_many_word(learn_learning_rate_scheduling_exponential,
        #                        objective_learning_rate_scheduling_exponential)
        # print_res(gd, points[i], max_iterations, "learning_rate_scheduling_exponential")
        # gd = run_too_many_word(learn_armijo, objective_armijo)
        # print_res(gd, points[i], max_iterations, "armijo")
        # gd = run_too_many_word(learn_wolfe, objective_wolfe)
        # print_res(gd, points[i], max_iterations, "wolfe")
        # gd = run_too_many_word(learn_bfgs, objective_bfgs)
        # print_res(gd, points[i], max_iterations, "bfgs")
        # gd_scipy = GradientDescentWithSciPy(func=sympy_func[i], method="BFGS")
        # print_res(gd_scipy, points[i], max_iterations, "scipy BFGS")
        # gd_scipy = GradientDescentWithSciPy(func=sympy_func[i], method="Newton-CG")
        # print_res(gd_scipy, points[i], max_iterations, "scipy Newton-CG")
        nt = run_too_many_word(learn_newton, objective_newton)
        print_r(nt(points[i], [4, 4]), "Newton")
        # print("------------------------------------------------------------------------\n\n\n")

    # rng = 5
    # show(func_table[3][0], rng, rng/20, 300, gd_scipy(startPoint=test_point, iterations=iter_max)[0])
