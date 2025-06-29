import math

import numpy as np

from lab1.Examples import func_table
from lab1.GradientDescent import GradientDescent, GDException
from lab1.OneDimensional import Wolfe


def get_model(dot_result, gradient, hess):
    return lambda p: dot_result + gradient @ p + (p.transpose() @ hess @ p) / 2


def find_minimum(mop, gd):
    return gd.next_point(gd.history()[-1], mop.learning_rate(gd))


def find_intersect(fromDot, path, sphereRadius):  # need to check
    a = np.dot(path, path)
    if a == 0:
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


def dog_leg(C_dot, delta, model, model_gradient_matrix_function, mop):
    gd = GradientDescent(model, model_gradient_matrix_function, mop, None)
    gd.vector = -model_gradient_matrix_function(np.array([0] * len(C_dot)).astype(np.longdouble))
    gd.history().append(np.array([0, 0]).astype(np.longdouble))
    if delta >= np.linalg.norm(C_dot):
        return C_dot
    B_dot = find_minimum(gd.learningRateCalculator, gd)
    if delta <= np.linalg.norm(B_dot):
        return B_dot / (np.linalg.norm(B_dot) / delta)
    A_dot = find_intersect(fromDot=B_dot, path=-B_dot + C_dot, sphereRadius=delta)
    return A_dot


hessCalculation = 0
hessDict = dict()


def newtoneMethodStart(
        function,
        gradient_matrix_function,  # function gives a gradient in the dot i provide
        hess_matrix_function,  # function gives a hess in the dot i provide
        x0,  # точка в пространстве начальная
        x1,  # ещё одна точка в пространстве, отличная от начальной больше, чем на минимум
        alpha_0,
        c1,
        c2,
        delta=1,  # начальная дельта
        iteration_stop_limit=0.000001,  # accuracy
        max_iter: int = 100_000_000,
        learning_rate=1,
        trust_upper_bound=3 / 4,
        trust_lower_bound=1 / 4,
        trust_no_trust_bound=1 / 16,
        trust_changing_multiply_value=2):
    assert max_iter > 0
    assert iteration_stop_limit > 0
    assert delta > 0

    x0 = np.array(x0).astype(np.longdouble)
    x1 = np.array(x1).astype(np.longdouble)

    mop = Wolfe(alpha_0, c1, c2, iteration_stop_limit)

    gd = GradientDescent(function, gradient_matrix_function, mop, None)

    global hessCalculation, hessDict
    hessCalculation = 0
    hessDict = dict()

    def hessF(x_k):
        global hessCalculation, hessDict
        x_k = tuple(x_k)
        if x_k in hessDict:
            return hessDict[x_k]
        hessCalculation += 1
        h = hess_matrix_function(x_k)
        h = np.array(h)
        hessDict[x_k] = h
        return h

    cur_iter_number = 0
    prev_x = x0
    cur_x = x1
    cur_result = gd.func(cur_x)

    # assert iteration_stop_limit < np.linalg.norm(cur_x - prev_x)
    while iteration_stop_limit > np.linalg.norm(cur_x - prev_x):
        prev_x += np.array([1, 1]) * iteration_stop_limit

    while np.linalg.norm(cur_x - prev_x) > iteration_stop_limit and max_iter > cur_iter_number:
        cur_result = gd.func(cur_x)
        gradient = gd.grad(cur_x)
        hess = hessF(cur_x)
        hess_reversed = np.linalg.inv(hess.astype(np.float64))
        # Вместо этого решать систему линейных уравнений + поддержка np.longdouble
        model = get_model(cur_result, gradient, hess)
        C_dot = - hess_reversed @ gradient
        is_trusted = False
        model_gradient_matrix_function = lambda p: gradient + hess @ p

        while not is_trusted:
            gd.vector = -gradient
            A_dot = dog_leg(C_dot, delta, model, model_gradient_matrix_function, mop)
            poss_result = gd.func(A_dot + cur_x)
            if (poss_result == cur_result):
                return [[cur_x] * cur_iter_number, gd.__funcCalculation__, gd.__gradCalculation__, hessCalculation]
            p_k = (cur_result - gd.func(A_dot + cur_x) + iteration_stop_limit / 16) / (
                    cur_result - model(A_dot) + iteration_stop_limit / 16)
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
            cur_x = A_dot + cur_x
            for i in cur_x:
                if math.isnan(i):
                    raise GDException()
        cur_iter_number += 1
    return [[cur_x] * cur_iter_number, gd.__funcCalculation__, gd.__gradCalculation__, hessCalculation]


if __name__ == '__main__':
    x0 = np.array([0.0, 0])
    x1 = np.array([120, 4])
    funcc = func_table[3][0]
    gradd = func_table[3][1]
    gess = func_table[3][2]
    print(newtoneMethodStart(
        function=funcc,
        gradient_matrix_function=gradd,
        hess_matrix_function=gess,
        x0=x0,
        x1=x1,
        alpha_0=12,
        c1=0.001,
        c2=0.01
    ))
