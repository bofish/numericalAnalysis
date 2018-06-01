import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LU_decomposition_example import LU_solver

if __name__ == '__main__':
    #----  Exact solution ----#
    n = 500
    x = np.linspace(0.0, 1.0, n)
    c1 = -1/4
    c2 = (1 + np.exp(-2)/4)/np.exp(-2)
    y_exact = c1*np.exp(-2*x) + c2*x*np.exp(-2*x) - x/4 + 1/4 

    #---- finite diif ----#
    # Step 1: Discretize x
    # n = 100
    # x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    y0 = 0
    y1 = 1

    # Step 2: Discretize ODE by 2nd order central difference scheme
    #         to form ODE into $a_iy_{i-1} + b_iy_i + c_iy_{i+1} = d_i$
    a_i = 1/dx**2 - 2/dx
    b_i = -2/dx**2 + 4
    c_i = 1/dx**2 + 2/dx
    d = [[x_i] if x_i!=x[n-2] else [x_i - c_i] for x_i in x[1:n-1]]

    # Step 6: Set up the linear system, return a banded structure matrix
    # Create a zero matrix A with n*n-2 dimension and zero vector b with n-2 dimension
    A = np.array([[0.0] * n for i in range(n-2)])

    # Perform banded strucuture iteration
    for i in range(n-2):
        row = i
        A[row, i] = a_i
        A[row, i+1] = b_i
        A[row, i+2] = c_i

    # Eliminate first and last column into a square matrix
    A = np.delete(A, n-1, 1)
    A = np.delete(A, 0, 1)

    # Solve the linear system of equations Ay = d
    y = LU_solver(A, d).tolist()
    y = np.array([[y0]] + y + [[y1]]).reshape(n)
    # print(y)

    plt.figure()
    plt.plot(x, y_exact, 'k',label='Exact solution')
    plt.plot(x, y, ':r', label='Finite difference solution')
    plt.legend()
    plt.show()