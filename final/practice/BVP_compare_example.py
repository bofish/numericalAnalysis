import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LU_decomposition_example import LU_solver
from IPV_2nd_order_example import RK4, forward_Euler

def exact_solution(x):
    c1 = -1/4
    c2 = (1 + np.exp(-2)/4)/np.exp(-2)
    y = c1*np.exp(-2*x) + c2*x*np.exp(-2*x) - x/4 + 1/4 
    return y

def finite_diff_solution(x):
    # Step 1: Discretize x
    n = len(x)
    dx = x[1] - x[0]
    y0 = 0.0
    y1 = 1.0

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
    return y

def shooting_solution(x):
    n = len(x)
    t0 = 0.0
    t1 = 1.0

    # The frist IVP
    def u_prime(t, u, g):
        return g
    def g_prime(t, u, g):
        return -4*g - 4*u + t

    u0 = 0.0
    g0 = 0.0
    vt, vu, vg = RK4(u_prime, g_prime, t0, u0, g0, t1, n)

    # The second IVP
    def v_prime(t, v, h):
        return h
    def h_prime(t, v, h):
        return -4*h - 4*v

    v0 = 0.0
    h0 = 1.0
    vt, vv, vh = RK4(v_prime, h_prime, t0, v0, h0, t1, n)

    # Combination
    beta = 1.0
    t_b = vt[-1]
    u_b = vu[-1]
    v_b = vv[-1]
    C = (beta - u_b)/v_b
    vx = vu + C*np.array(vv)
    return vt, vx

if __name__ == '__main__':
    n = 101
    x = np.linspace(0.0, 1.0, n)

    #----  Exact solution ----#
    y_exact = exact_solution(x)

    #---- finite diif ----#
    y_fd = finite_diff_solution(x)

    #---- Shooting method solution ----#
    x_sh, y_sh = shooting_solution(x)

    plt.figure()
    plt.plot(x, y_exact, 'k',label='Exact solution')
    plt.plot(x, y_fd, ':r', label='Finite difference solution')
    plt.plot(x_sh, y_sh, '-.b', label='Shooting method solution')
    plt.legend()
    plt.grid()
    plt.show()