import math
import random
import numpy as np


class Genetic:
    def fitness(self, x):
        return self.func(x)

    def generate_random_from_borders(self):
        x = []
        dims = len(self.borders)
        for j in range(dims):
            x.append(random.randrange(self.borders[j][0], self.borders[j][1]))
        return np.array(x).astype(np.longdouble)

    def generate(self, cur):
        n = len(cur)
        cur.sort(key = lambda x: self.fitness(x))
        pars = cur[0:math.ceil(math.sqrt(n))]
        random.shuffle(pars)
        news = []
        for i in range(n):
            news.append(self.get_child(random.choice(pars), random.choice(pars)))
        return news

    def get_child(self, p1, p2):
        alpha = 0.5
        ch = alpha * p1 + (1.0 - alpha) * p2
        return self.get_mutation(ch)

    def get_mutation(self, x):
        r = self.generate_random_from_borders()
        x = self.mutation_rate * r + x * (1 - self.mutation_rate)
        return x

    def func(self, x):
        x = tuple(x)
        if x in self.func_dic:
            return self.func_dic[x]
        f = self.function(x)
        self.func_dic[x] = f
        self.func_calcs += 1
        return f

    def run(self, func, steps, borders, generation_size=100, mutation_rate=0.1):
        self.function = func
        self.func_calcs = 0
        self.func_dic = dict()
        self.borders = borders
        self.mutation_rate = mutation_rate
        generation = []
        for i in range(generation_size):
            generation.append(self.generate_random_from_borders())
        for _ in range(steps):
            generation = self.generate(generation)
        generation.sort(key = lambda x: self.fitness(x))
        return generation[0], self.func_calcs