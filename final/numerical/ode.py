from math import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numerical import taylor

def RK4(f, g, t0, y0, v0, t1, N):
    vt = [0] * N
    vy = [0] * N
    vv = [0] * N

    dt = (t1 - t0) / float(N-1)
    vt[0] = t = t0
    vy[0] = y = y0
    vv[0] = v = v0
    for i in range(1, N):
        k1 = dt * f(t, y, v)
        l1 = dt * g(t, y, v)
        k2 = dt * f(t + 0.5*dt, y + 0.5*k1, v + 0.5*l1)
        l2 = dt * g(t + 0.5*dt, y + 0.5*k1, v + 0.5*l1)
        k3 = dt * f(t + 0.5*dt, y + 0.5*k2, v + 0.5*l2)
        l3 = dt * g(t + 0.5*dt, y + 0.5*k2, v + 0.5*l2)
        k4 = dt * f(t + dt, y + k3, v + l3)
        l4 = dt * g(t + dt, y + k3, v + l3)
        vt[i] = t = t0 + i * dt
        vy[i] = y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        vv[i] = v = v + (l1 + 2*l2 + 2*l3 + l4) / 6
    return vt, vy, vv

def forward_Euler(f, g, t0, y0, v0, t1, N):
    vt = [0] * N
    vy = [0] * N
    vv = [0] * N

    dt = (t1 - t0) / float(N-1)
    vt[0] = t = t0
    vy[0] = y = y0
    vv[0] = v = v0
    for i in range(1, N):
        vt[i] = t = t0 + i * dt
        vy[i] = y = y + dt * f(t, y, v)
        vv[i] = v = v + dt * g(t, y, v)
    return vt, vy, vv

def RK4_2d(dW, W0, N, dt, Nt, Re):
    t0 = 0
    vt = np.zeros(Nt)
    vW = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    vU = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    vV = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])

    vt[0] = t = t0
    vW[0] = W = W0
    (vU[0], vV[0]) = (U, V) = taylor.get_U_V(W0)
    
    for i in range(1, Nt):
        # print(i)
        k1 = dt * dW(W, Re)
        k2 = dt * dW(W + 0.5*k1, Re)
        k3 = dt * dW(W + 0.5*k2, Re)
        k4 = dt * dW(W + k3, Re)
        vt[i] = t = t0 + i * dt
        vW[i] = W = W + (k1 + 2*k2 + 2*k3 + k4) / 6
        (vU[i], vV[i]) = (U, V) = taylor.get_U_V(W)
    dt_max = taylor.monitor_time_stability(vU, vV, dt, Re, N)
    return vt, vW, dt_max

if __name__ == '__main__':

    def f(t, y, v):
        return v
    def g(t, y, v):
        return -v + np.sin(t*y)
    N = 101
    t0 = 0
    t1 = 5.0
    y0 = 1
    v0 = 2

    # RK4
    vt, vy, vv = RK4(f, g, t0, y0, v0, t1, N)
    # Forward Euler
    vt_E, vy_E, vv_E = forward_Euler(f, g, t0, y0, v0, t1, N)
    # Exact solution
    def pend(Y, t):
        y, v = Y
        dYdt = [v, -v+np.sin(t*y)]
        return dYdt

    Y0 = [1, 2]
    times = np.linspace(0, 5.0, 101)
    sol = odeint(pend, Y0, times)

    plt.figure()
    plt.plot(times, sol[:, 0], 'k', label='Exact')
    plt.plot(vt, vy, ':r', label='RK4')
    plt.plot(vt_E, vy_E, '-.b', label='Forward Euler')
    plt.legend()
    plt.show()