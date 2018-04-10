import numpy as np
from numpy.linalg import lstsq
from math import pi, cos
import matplotlib.pyplot as plt
from functools import reduce

def prod(iterable):
    '''
    https://stackoverflow.com/questions/595374/whats-the-python-function-like-sum-but-for-multiplication-product
    >>> prod(range(1, 5))
    24
    '''
    return reduce(operator.mul, iterable, 1)

def origin_fun(a, b):
    x = np.linspace(a, b, 100)
    f = np.cos(10*x**2)*np.exp(-x)

    return x, f

def evenly_space_pt(a,b,N):
    x = np.linspace(a, b, N)
    f = np.cos(10*x**2)*np.exp(-x)

    return x, f

def chebyshev_pt(a, b, N):
    x = np.array([(a+b)/2 - (b-a)/2*cos(k*pi) for k in np.linspace(0,1,N)])
    f = np.cos(10*x**2)*np.exp(-x)

    return x, f

def polynomial_vandermode(x, f, x_sub):
    assert len(x)==10, 'only support N=10'
    correspond_mat = np.matrix([
        [1, x[0], x[0]**2, x[0]**3, x[0]**4, x[0]**5, x[0]**6, x[0]**7, x[0]**8, x[0]**9],
        [1, x[1], x[1]**2, x[1]**3, x[1]**4, x[1]**5, x[1]**6, x[1]**7, x[1]**8, x[1]**9],
        [1, x[2], x[2]**2, x[2]**3, x[2]**4, x[2]**5, x[2]**6, x[2]**7, x[2]**8, x[2]**9],
        [1, x[3], x[3]**2, x[3]**3, x[3]**4, x[3]**5, x[3]**6, x[3]**7, x[3]**8, x[3]**9],
        [1, x[4], x[4]**2, x[4]**3, x[4]**4, x[4]**5, x[4]**6, x[4]**7, x[4]**8, x[4]**9],
        [1, x[5], x[5]**2, x[5]**3, x[5]**4, x[5]**5, x[5]**6, x[5]**7, x[5]**8, x[5]**9],
        [1, x[6], x[6]**2, x[6]**3, x[6]**4, x[6]**5, x[6]**6, x[6]**7, x[6]**8, x[6]**9],
        [1, x[7], x[7]**2, x[7]**3, x[7]**4, x[7]**5, x[7]**6, x[7]**7, x[7]**8, x[7]**9],
        [1, x[8], x[8]**2, x[8]**3, x[8]**4, x[8]**5, x[8]**6, x[8]**7, x[8]**8, x[8]**9],
        [1, x[9], x[9]**2, x[9]**3, x[9]**4, x[9]**5, x[9]**6, x[9]**7, x[9]**8, x[9]**9]
    ])
    forcing_term = np.matrix([
        [f[0]],
        [f[1]],
        [f[2]],
        [f[3]],
        [f[4]],
        [f[5]],
        [f[6]],
        [f[7]],
        [f[8]],
        [f[9]]
    ])
    sol = lstsq(correspond_mat, forcing_term, rcond=-1)
    coeff = sol[0]
    f_hat = coeff.item(0)*np.ones(len(x_sub)) + coeff.item(1)*x_sub + coeff.item(2)*x_sub**2 + coeff.item(3)*x_sub**3 + coeff.item(4)*x_sub**4 + coeff.item(5)*x_sub**5 + coeff.item(6)*x_sub**6 + coeff.item(7)*x_sub**7 + coeff.item(8)*x_sub**8 + coeff.item(9)*x_sub**9
    return f_hat

def polynomial_lagrange(x, f):
    M = len(x)
    poly = np.poly1d(0.0)
    for j in range(M):
        single_item = np.poly1d(f[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j] - x[k]
            single_item *= np.poly1d([1.0, -x[k]])/fac
        poly += single_item
    
    return poly

def polynomial_newton(x, f):
    M = len(x)
    correspond_mat = np.matrix([
        [1]
    ])
    for j in range(M):
        single_item = 1
        for k in range(j):

            fac = x[j] - x[k]
            single_item *= fac
        poly += single_item
    
    return poly


if __name__ == '__main__':

    N = 10
    a = -1
    b = 1

    print('-Evenly spaced points-')
    x_o, f_o = origin_fun(a,b)

    print('-Evenly spaced points-')
    x_es, f_es = evenly_space_pt(a, b, N)

    # print('\n'.join('{0:2d} & {1:4.3f} & {2:5.4f}  \\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))

    print('-Chebyshev points-')

    x_cp, f_cp = chebyshev_pt(a, b, N)

    # print('\n'.join('{0:2d} & {1:4.3f} & {2:5.4f}  \\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))
    x_sub = np.linspace(-1,1,100)
    f_val = polynomial_vandermode(x_cp, f_cp, x_sub)

    # plt.plot(x_o, f_o, 'k', x_cp, f_cp, 'or', x_sub, f_val, 'b')
    # plt.show()

    lpoly = polynomial_lagrange(x_es, f_es)
    plt.plot(x_o, f_o, 'k', x_es, f_es, 'or', x_sub, lpoly(x_sub), 'b')
    plt.show()