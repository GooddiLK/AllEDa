from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import math
import numpy as np
from numpy import arange
from numpy import meshgrid


def to2(func):
    return lambda x, y: func([x, y])


def show(func, rng, grid, last_points, r):
    if last_points > 0:
        r = r[-last_points:]
    lx, ly = r[-1]
    xaxis = arange(-rng + lx, rng + lx, grid)
    yaxis = arange(-rng + ly, rng + ly, grid)
    x, y = meshgrid(xaxis, yaxis)
    results = to2(func)(x, y)
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(x, y, results, cmap='viridis', alpha=0.5)
    rx = np.array([i[0] for i in r])
    ry = np.array([i[1] for i in r])
    rz = to2(func)(rx, ry)
    indices = np.linspace(0, 1, len(r))
    indices = np.array([math.exp(-i) for i in indices])
    colors = ["green", "red"]
    cmap_custom = LinearSegmentedColormap.from_list("RedToGreen", colors)
    axis.scatter(
        rx, ry, rz,
        c=indices,
        cmap=cmap_custom,
    )
    plt.show()