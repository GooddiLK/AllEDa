import math
import random


class Annealing:
    def calc_new(self, cur, t, x_new):
        v = self.func(cur)
        v_new = self.func(x_new)
        df = v_new - v
        if v_new < v:
            return x_new
        print(f"cur:{cur} == {v}\nnew{x_new} == {v_new}")
        if random.random() < math.exp(-df/t):
            print("прок")
            return x_new
        print("не прок")
        return cur

    def func(self, x):
        x = tuple(x)
        if x in self.func_dic:
            return self.func_dic[x]
        f = self.function(x)
        self.func_dic[x] = f
        self.func_calcs += 1
        return f

    def run(self, cur, func, t_max=100, t_min=1, alpha=0.95):
        t = t_max
        self.function = func
        self.func_calcs = 0
        self.func_dic = dict()
        while t > t_min:
            x_new = [xi + (random.random() - 0.5) * t for xi in cur]
            cur = self.calc_new(cur, t, x_new)
            t = alpha * t
        return cur

