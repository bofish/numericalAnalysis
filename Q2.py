import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from scipy.special import erf

#----Q2(a)----#
a = 2/pi
N = 2**10
x = np.linspace(-pi, pi, N)
g = np.exp(-a*x**2)


# G analytical
s = np.linspace(-1000, 1000, N)
G_analytical = sqrt(pi)*np.exp(-pi**2*s**2/a**2)
plt.figure()
plt.plot(s, G_analytical, label='G(s) (Analytical Expression)')

# f series analytical
x = np.linspace(-5*pi, 5*pi, N)
f_analytical = []
for x_m in x:
    f_m = []
    for m in np.arange(-500,500):
        f_m.append(np.exp(-a*(x_m-2.0*pi*m)**2))
    f_analytical.append(np.sum(f_m))

plt.figure()
plt.plot(x, f_analytical, label='f(x) (Analytical Expression)')
plt.plot(x, g, label='g(x)')

# f_hat analytical
f_hat_approximation = []
for n in range(-10,10):
    x_u = 2*pi
    x_l = 0
    integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
    f_hat_approximation.append( np.exp(-n**2/(4*a))*sqrt(pi)*integ_val/(4*pi*sqrt(a)) )

f_fourier_analytical = []
for x_n in x:
    sum_temp = []
    for n in range(-10, 10):
        sum_temp.append( f_hat_approximation[n+10]*np.exp(1j*n*x_n) )
    f_fourier_analytical.append(np.sum(sum_temp))
# print(len(f_fourier_analytical))
# plt.figure()
plt.plot(x, f_fourier_analytical)

plt.legend()
plt.show()