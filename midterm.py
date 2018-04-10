import timeit
import numpy as np
from math import pi, cos
N = 10
a = -1
b =  1

print('-Evenly spaced points-')
x = np.linspace(a, b, N)
f = np.cos(10*x**2)*np.exp(-x)

print('\n'.join('{0} & {1:4.3f} & {2:5.4f}\\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))

print('-Chebyshev points-')


x = np.array([(a+b)/2 - (b-a)/2*cos(k*pi) for k in np.linspace(0,1,N)])
f = np.cos(10*x**2)*np.exp(-x)

print('\n'.join('{0:2d} & {1:4.3f} & {2:5.4f}\\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))