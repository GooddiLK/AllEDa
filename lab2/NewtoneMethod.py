import numpy as np


def get_model(function, gradient_matrix_function, hess_matrix_function, xk):
    return lambda p: function(xk) + gradient_matrix_function(xk) @ p + (p.transpose @ hess_matrix_function(xk) @ p) / 2


def newtoneMethodStart(function, gradient_matrix_function, hess_matrix_function, x0=0, delta=1,
                       iteration_stop_limit=1e-5, max_iter=100_000, learning_rate=1):
    assert max_iter > 0
    assert iteration_stop_limit != 0
    assert delta > 0

    cur_iter_number = 0

    prev_x = x0
    prev_result = function(prev_x)
    cur_x = x0 + 1
    cur_result = function(cur_x)

    assert iteration_stop_limit < cur_x - prev_x
    while cur_x - prev_x < iteration_stop_limit and max_iter < cur_iter_number:
        model = get_model(function, gradient_matrix_function, hess_matrix_function, cur_x)
