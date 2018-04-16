import timeit
import numpy as np
import numpy.matlib
from numpy.linalg import lstsq
from math import pi, cos
from functools import reduce
import matplotlib.pyplot as plt


x = np.linspace(0,10,5)
f = x/3

x_sub = np.linspace(0,10,10)


def prod(iterable):

    return reduce(lambda x, y: x*y, iterable, 1) 

def polynomial_newton(x, f):
    M = len(x)
    # Calculate Coefficient
    c = [f[0]]
    for i in range(1,M):
        fac = 0
        for j in range(1,i):
            fac += c[j]*prod(x[i] - x[0:j])
        c.append((f[i] - c[0] - fac)/prod(x[i] - x[0:i]))
    
    # Obtain polynomial
    poly = np.poly1d(c[0])
    for j in range(1,M):
        single_item = np.poly1d(1.0)
        for k in range(0,j):
            single_item *= np.poly1d([1.0, -x[k]])
        poly += c[j]*single_item
    return poly

def nature_spline_coeff(x, f):
    M = len(x)
    delta = []
    delta_f = []
    coeff_mat = np.matlib.zeros((M-2, M))
    f_mat = np.matlib.zeros((M-2, 1))
    for j in range(1,M):
        delta.append( x[j] - x[j-1])
        delta_f.append((f[j] - f[j-1])/(x[j] - x[j-1]))
    for j in range(1,M-1):
        row = j-1
        coeff_mat[row, j-1] = delta[j-1]/6
        coeff_mat[row, j] = (delta[j-1]+delta[j])/3
        coeff_mat[row, j+1] = delta[j]/6
        f_mat[row, 0] = delta_f[j] - delta_f[j-1]
    # For the nature spline, we given the conditin which is f1" = fn" = 0
    coeff_mat = np.delete(coeff_mat, M-1, 1)
    coeff_mat = np.delete(coeff_mat, 0, 1)

    res = lstsq(coeff_mat, f_mat, rcond=-1)
    ddF = [0]
    for i in range(res[2]):
        ddF.append(res[0].item(i))
    ddF.append(0)
    return ddF, delta, delta_f

def cubic_spline(x, f, sub_xs):
    M = len(x)
    (ddF, delta, delta_f) = nature_spline_coeff(x, f)
    f_vals = []
    for sub_x in sub_xs:
        for j in range(M-1):
            if sub_x <= x[j+1]:
                f_val = ((x[j+1] - sub_x)**3/(6*delta[j]) - delta[j]*(x[j+1] - sub_x)/6)*ddF[j] \
                + ((sub_x - x[j])**3/(6*delta[j]) - delta[j]*(sub_x - x[j])/6)*ddF[j+1] + delta_f[j]*(sub_x - x[j]) + f[j]
                f_vals.append(f_val)
                break
            else:
                pass
            
    return f_vals

# x = np.linspace(-pi,pi,4)
# f = np.sin(x)
# x_sub = np.linspace(-pi,pi,100)
# poly = polynomial_newton(x,f)
# plt.plot(x,f,'ob',x_sub,poly(x_sub),'-r')
# plt.show()
x = np.array([0, 2, 3, 5, 9, 15])
f = np.array([32, 12, 43, 55, 66, 22])
x_sub = np.linspace(0,15,50)

f_val = cubic_spline(x, f, x_sub)
plt.plot(x, f, 'ob', x_sub, f_val, '-r')
plt.show()
# print(delta)