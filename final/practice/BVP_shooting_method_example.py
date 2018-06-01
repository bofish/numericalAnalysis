import numpy as np
import matplotlib.pylab as plt
from IPV_2nd_order_example import RK4, forward_Euler

if __name__ == '__main__':
    #----  Exact solution ----#
    n = 500
    x = np.linspace(0.0, 1.0, n)
    c1 = -1/4
    c2 = (1 + np.exp(-2)/4)/np.exp(-2)
    y_exact = c1*np.exp(-2*x) + c2*x*np.exp(-2*x) - x/4 + 1/4 

    #---- Shooting method solution ----#
    N = 101
    t0 = 0.0
    t1 = 1.0

    # The frist IVP
    def u_prime(t, u, g):
        return g
    def g_prime(t, u, g):
        return -4*g - 4*u + t

    u0 = 0.0
    g0 = 0.0
    vt, vu, vg = RK4(u_prime, g_prime, t0, u0, g0, t1, N)

    # The second IVP
    def v_prime(t, v, h):
        return h
    def h_prime(t, v, h):
        return -4*h - 4*v

    v0 = 0.0
    h0 = 1.0
    vt, vv, vh = RK4(v_prime, h_prime, t0, v0, h0, t1, N)

    # Combination
    beta = 1.0
    t_b = vt[-1]
    u_b = vu[-1]
    v_b = vv[-1]
    C = (beta - u_b)/v_b
    vx = vu + C*np.array(vv)

    plt.figure()
    plt.plot(x, y_exact, 'k', label='Exact solution')
    plt.plot(vt, vx, ':r', label='Shooting method solution')
    plt.legend()
    plt.grid()
    plt.show()