import numpy as np
import matplotlib.pylab as plt
from IPV_2nd_order_example import RK4, forward_Euler

if __name__ == '__main__':

    # Apply shooting method
    N = 101
    t0 = 0.0
    t1 = 4.0

    # The frist IVP
    def u_prime(t, u, g):
        return g
    def g_prime(t, u, g):
        return 2*t/(1+t**2)*g - 2/(1+t**2)*u + 1

    u0 = 1.25
    g0 = 0.0
    vt, vu, vg = RK4(u_prime, g_prime, t0, u0, g0, t1, N)

    # The second IVP
    def v_prime(t, v, h):
        return h
    def h_prime(t, v, h):
        return 2*t/(1+t**2)*h - 2/(1+t**2)*v

    v0 = 0.0
    h0 = 1.0
    vt, vv, vh = RK4(v_prime, h_prime, t0, v0, h0, t1, N)

    # Combination
    beta = -0.95
    t_b = vt[-1]
    u_b = vu[-1]
    v_b = vv[-1]
    C = (beta - u_b)/v_b
    vx = vu + C*np.array(vv)

    plt.figure()
    plt.plot(vt, vx, 'k', label='x(t)')
    plt.plot(vt, vv, ':r', label='v(t)')
    plt.plot(vt, vu, '-.b', label='u(t)')
    plt.legend()
    plt.grid()
    plt.show()