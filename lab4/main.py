from lab1.Examples import func_table, reals
from lab2.Optuna import min_norm
from lab4.Annealing import Annealing
from lab4.Genetic import Genetic

if __name__ == "__main__":
    # Метод отжига
    # a = Annealing()
    # start = [(), (100, -200), (100, -200), (100, -200), (100, -200)]
    # for i in range(1, 5):
    #     print(i)
    #     print("----------------------")
    #     find, calcs = a.run(start[i], func_table[i][0], 100_000_000, 0.001, 0.99)
    #     norm = min_norm(find, reals[i])
    #     print(find, calcs, norm)
    # Генетический алгоритм
    a = Genetic()
    borders = [[], [[-100, 50], [-100, 50]], [[-100, 50], [-100, 50]], [[-100, 50], [-100, 50]], [[-100, 50], [-100, 50]]]
    for i in range(1, 5):
        print(i)
        print("----------------------")
        find, calcs = a.run(func_table[i][0], 10, borders[i], 100, 0.005)
        norm = min_norm(find, reals[i])
        print(find, calcs, norm)
