import numpy as np


def proizv(function, x, prev_x, result, prev_result):
    return (x - prev_x) / (result - prev_result)


def newtoneMethodStart(function, x0=0, step=1, iteration_stop_limit=1e-5, max_iter=100_000):
    prev_x = x0
    prev_result = function(prev_x)
    cur_x = x0/2 + 1
    cur_result = function(cur_x)
    while True:
        next_x = cur_x - function(x0) / (proizv(function, cur_x, prev_x, cur_result, prev_result)(x0))
