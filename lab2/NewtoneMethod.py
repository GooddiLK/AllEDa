import numpy as np

from lab1.GradientDescent import GradientDescent
from lab1.OneDimensional import Wolfe


def get_model(dot_result, gradient, hess):
    return lambda p: dot_result + gradient @ p + (p.transpose @ hess @ p) / 2


def find_minimum(mop, gd):
    return gd.next_point(gd.history()[-1], mop.learning_rate(gd))


def find_intersect(fromDot, path, sphereRadius):  # need to check
    a = np.dot(path, path)
    if a == 0:
        if np.linalg.norm(fromDot) == sphereRadius:
            return fromDot
        else:
            return fromDot

    b = 2 * np.dot(fromDot, path)
    c = np.dot(fromDot, fromDot) - sphereRadius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        t_min = - (np.dot(fromDot, path)) / a
        t_clipped = np.clip(t_min, 0.0, 1.0)
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        valid_ts = [t for t in [t1, t2] if 0 <= t <= 1]
        if valid_ts:
            t_clipped = min(valid_ts)
        else:
            norm_from = np.linalg.norm(fromDot)
            norm_to = np.linalg.norm(fromDot + path)
            if abs(norm_from - sphereRadius) <= abs(norm_to - sphereRadius):
                t_clipped = 0.0
            else:
                t_clipped = 1.0
    return fromDot + t_clipped * path


def dog_leg(function, xk, C_dot, gradient, delta, gradient_matrix_function):
    mop = Wolfe(12, 0.001, 0.1, 0.0001)
    gd = GradientDescent(function, gradient_matrix_function, mop, None)
    gd.vector = -gradient
    gd.history().append(xk.astype(np.longdouble))

    B_dot = find_minimum(mop, gd) - xk
    A_dot = find_intersect(fromDot=B_dot, path=-B_dot + C_dot, sphereRadius=delta)
    return A_dot


def newtoneMethodStart(
        function,
        gradient_matrix_function,  # function gives a gradient in the dot i provide
        hess_matrix_function,  # function gives a hess in the dot i provide
        x0,  # точка в пространстве начальная
        x1,  # ещё одна точка в пространстве, отличная от начальной больше, чем на минимум
        delta=1,  # начальная дельта
        iteration_stop_limit=0.001,
        max_iter: int = 100_000,
        learning_rate=1,
        trust_upper_bound=3 / 4,
        trust_lower_bound=1 / 4,
        trust_no_trust_bound=1 / 16,
        trust_changing_multiply_value=2):
    assert max_iter > 0
    assert iteration_stop_limit > 0
    assert delta > 0

    cur_iter_number = 0
    prev_x = x0
    cur_x = x1
    cur_result = function(cur_x)

    assert iteration_stop_limit < np.linalg.norm(cur_x - prev_x)
    while np.linalg.norm(cur_x - prev_x) < iteration_stop_limit and max_iter > cur_iter_number:
        gradient = gradient_matrix_function(cur_x)
        hess = hess_matrix_function(cur_x)
        hess_reversed = hess.__invert__()  # переделать потом можно -- вместо этого решать систему линейных уравнений
        model = get_model(cur_result, gradient, hess)
        p = -hess_reversed @ gradient
        C_dot = learning_rate * p
        is_trusted = False

        while not is_trusted:
            A_dot = dog_leg(function, cur_x, C_dot, gradient, delta, gradient_matrix_function)
            p_k = (cur_result - function(A_dot + cur_x)) / (
                    cur_result - model(A_dot))
            if p_k > trust_upper_bound:
                delta *= trust_changing_multiply_value
            elif p_k > trust_lower_bound:
                delta = delta
            elif p_k > trust_no_trust_bound:
                delta /= trust_changing_multiply_value
            else:
                delta /= trust_changing_multiply_value
                is_trusted = False
                continue
            is_trusted = True
            prev_x = cur_x
            cur_x = A_dot
        cur_iter_number += 1


x0 = np.array([0, 0])
x1 = np.array([0, 1])
newtoneMethodStart(
    function=lambda vector: vector[0] ^ 2 + vector[1] ^ 2,
    gradient_matrix_function=lambda vector: np.array([2 * vector[0], 2 * vector[1]]),
    hess_matrix_function=lambda vector: np.array([2, 0], [0, 2]),
    x0=x0,
    x1=x1,
)
