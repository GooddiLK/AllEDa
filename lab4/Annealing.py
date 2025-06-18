import math
import random


class Annealing:
    def calc_new(self, cur, t, x_new):
        v = self.func(cur)
        v_new = self.func(x_new)
        df = v_new - v
        if v_new < v or random.random() < math.exp(-df/t):
            return v_new
        return v

    def func(self, x):
        if x in self.func_dic:
            return self.func_dic[x]
        f = self.function(x)
        self.func_dic[x] = f
        self.func_calcs += 1
        return f

    def run(self, cur, func, t_max=1000, t_min=0.1, alpha=0.95):
        t = t_max
        self.function = func
        self.func_calcs = 0
        self.func_dic = dict()
        while t > t_min:
            x_new = [xi + random.random() * t for xi in cur]
            cur = self.calc_new(cur, t, x_new)
            t = alpha * t
        return cur

