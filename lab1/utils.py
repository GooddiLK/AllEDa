import sympy as sp

# Функция для вычисления градиента
def compute_gradient(func, variables):
    grad = [sp.diff(func, var) for var in variables]
    return sp.lambdify(variables, grad, "numpy")

# Функция для вычисления Гессиана
def compute_hessian(func, variables):
    hessian = [[sp.diff(sp.diff(func, var1), var2) for var2 in variables] for var1 in variables]
    return sp.lambdify(variables, hessian, "numpy")

def print_res_for_scipy(result):
    print(f"result: {result[0][-1]}")
    print(f"iterations: {len(result[0])}")
    print(f"func eval count: {result[1]}")
    print(f"grad eval count: {result[2]}")
    print(f"hess eval count: {result[3]}")

