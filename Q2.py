import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from scipy.special import erf

#----Q2(a)----#

# G analytical
s = np.linspace(-5, 5, N)
G_analytical = sqrt(pi)*np.exp(-pi**2*s**2/a**2)
plt.figure()
plt.plot(s, G_analytical, label='G(s) (Analytical Expression)')

# f function analytical
a = 2/pi
N = 2**10
x = np.linspace(-3*pi, 3*pi, N)
g = np.exp(-a*x**2)
f_analytical = []
for x_m in x:
    f_m = []
    for m in np.arange(-10,10):
        f_m.append(np.exp(-a*(x_m-2.0*pi*m)**2))
    f_analytical.append(np.sum(f_m))

# f_hat analytical
x_u = pi
x_l = -pi
f_fourier_analytical = []
for x_n in x:
    sum_temp = []
    for n in range(-10, 10):
        integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
        f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi)*integ_val/(4*pi*sqrt(a))
        sum_temp.append( f_hat_approximation*np.exp(1j*n*x_n) )
    f_fourier_analytical.append(np.sum(sum_temp))

plt.figure()
plt.plot(x, f_analytical, '-.', label='f(x) (Analytical Expression)')
plt.plot(x, g, '--',label='g(x)')
plt.plot(x, np.real(f_fourier_analytical), ':', label='f(x) (Foureir series)')

plt.legend()
plt.show()