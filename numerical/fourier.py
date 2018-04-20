from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

# # Number of samplepoints
# N = 600
# # sample spacing

# x = np.linspace(-10, 10, N)
# s = np.linspace(-10, 10, N)

# a = 1
# g = np.exp(-a*x**2)
# G_analytical = sqrt(pi)*np.exp(-pi**2*s**2/a**2)
# G_numerical = np.fft.fft(g)

# fig, ax = plt.subplots()
# ax.plot(s, G_analytical)

# fig2, ax2 = plt.subplots()
# ax2.plot(x, g)

# fig3, ax3 = plt.subplots()
# ax3.plot(x, G_numerical)
# plt.show()

from scipy.fftpack import fft
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.figure()
plt.plot(x, y)
plt.grid()
plt.show()