import timeit
import numpy as np
from math import pi, cos

x = np.linspace(0,10,5)
f = x/3

x_sub = np.linspace(0,10,10)


def prod(iterable):
    return reduce(lambda x, y: x*y, iterable, 1)