from lab1.Examples import func_table, reals
from lab2.Optuna import min_norm
from lab4.Annealing import Annealing

if __name__ == "__main__":
    a = Annealing()

    start = [(), (100, -200), (100, -200), (100, -200), (100, -200)]
    for i in range(1, 5):
        print(i)
        print("----------------------")
        find, calcs = a.run(start[i], func_table[i][0], 100000000, 0.001, 0.99)
        norm = min_norm(find, reals[i])
        print(find, calcs, norm)
