from lab1.Examples import func_table
from lab4.Annealing import Annealing

if __name__ == "__main__":
    a = Annealing()
    print(a.run([100, -100], func_table[4][0], 1000, 0.1, 0.99))