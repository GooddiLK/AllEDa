import numpy as np


def get_model(dot_result, gradient, hess):
    return lambda p: dot_result + gradient @ p + (p.transpose @ hess @ p) / 2


def dog_leg(model, C_dot, gradient, delta):
    B_dot = find_minimum()
    A_dot = find_intersect(fromDot=B_dot, path=-B_dot + C_dot, sphereRadius=delta)
    return A_dot


def newtoneMethodStart(
        function,
        gradient_matrix_function,
        hess_matrix_function,
        x0,
        x1,
        delta=1,
        iteration_stop_limit=1e-5,
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

    assert iteration_stop_limit < cur_x - prev_x
    while np.linalg.norm(cur_x - prev_x) < iteration_stop_limit and max_iter > cur_iter_number:
        gradient = gradient_matrix_function(cur_x)
        hess = hess_matrix_function(cur_x)
        hess_reversed = hess.__invert__()  # переделать потом можно -- вместо этого решать систему линейных уравнений
        model = get_model(cur_result, gradient, hess)
        p = -hess_reversed @ gradient
        C_dot = learning_rate * p
        is_trusted = False

        while not is_trusted:
            A_dot = dog_leg(model, C_dot, gradient, delta)
            p_k = (cur_result - function(A_dot + cur_x)) / (model(np.array([0]*m)) - model(A_dot - cur_x))  # m is power
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
