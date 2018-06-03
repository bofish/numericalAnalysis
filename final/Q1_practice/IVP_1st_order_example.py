import numpy as np
import matplotlib.pyplot as plt
N = 100
t = np.linspace(0, 5.0, N)
dt = t[1] - t[0]

yEU = [0]
yRK4 = [0]
# v = [2]

for t_k in t:
    yEU_k = yEU[-1]
    # v_k = v[-1]
    yEU_kp1 = yEU_k + dt*(-0.5*np.exp(t_k/2)*np.sin(5*t_k) + 5*np.exp(t_k/2)*np.cos(5*t_k) + yEU_k)
    # v_kp1 = v_k + dt*(-v_k + np.sin(t_k*y_k))
    yEU.append(yEU_kp1)
    # v.append(v_kp1)

    y_k = yRK4[-1]
    k1 = dt*(-0.5*np.exp(t_k/2)*np.sin(5*t_k) + 5*np.exp(t_k/2)*np.cos(5*t_k) + y_k)
    k2 = dt*(-0.5*np.exp((t_k+0.5*dt)/2)*np.sin(5*(t_k+0.5*dt)) + 5*np.exp((t_k+0.5*dt)/2)*np.cos(5*(t_k+0.5*dt)) + (y_k + 0.5*k1))
    k3 = dt*(-0.5*np.exp((t_k+0.5*dt)/2)*np.sin(5*(t_k+0.5*dt)) + 5*np.exp((t_k+0.5*dt)/2)*np.cos(5*(t_k+0.5*dt)) + (y_k + 0.5*k2))
    k4 = dt*(-0.5*np.exp((t_k+dt)/2)*np.sin(5*(t_k+dt)) + 5*np.exp((t_k+dt)/2)*np.cos(5*(t_k+dt)) + (y_k + k3))
    yRK4_kp1 = y_k + (k1 + 2*k2 + 2*k3 + k4)/6
    yRK4.append(yRK4_kp1)
    
times = np.linspace(0, 5.0, 1000)
y_anal = np.exp(times/2)*np.sin(5*times)
plt.figure()
plt.plot(times, y_anal, 'k')
plt.plot(t, yRK4[0:N], 'r')
plt.plot(t, yEU[0:N], 'b')
plt.show()
