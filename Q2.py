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

# ----Q2(b)----#
N = 2**7
a = 5
x = np.linspace(-pi, pi, N)
g = np.exp(-a*x**2)
g_tilde = FFT(g)/N
n = np.linspace(-N/2, N/2, N)
g_tilde_shift = FFTshift(g_tilde)

plt.figure()
plt.plot(n, abs(g_tilde_shift), label='g_tilde(a)')
plt.xlabel('n')
plt.ylabel('g_tilde(a)')
plt.title('N={}'.format(N))
plt.legend()

f_fourier = []  # Fourier series interpolation
for x_n in x:
    single_term = []
    for n in range(-N//2, N//2):
        single_term.append(g_tilde[n]*np.exp(1j*n*x_n))
    f_fourier.append(np.sum(single_term))
f_fourier = FFTshift(f_fourier)
plt.figure()
plt.plot(g, '-k', label='g(x)')
plt.plot(np.real(f_fourier), ':g', label='Sf(x)')
plt.xlabel('x')
plt.ylabel('function value')
plt.title('a={}, N={}'.format(a, N))
plt.legend()

#----Q2(c)----#
N = 2**10
a = 5
x = np.linspace(-2*pi, 2*pi, N)
s = np.linspace(-10, 10, N)
g = np.exp(-a*x**2)

# G analytical
G_s = cal_fourier_transform(s, a)

# f_hat
x_boundary = [-pi, pi]
N_f = 64 # the power of series, i.g. n = [-25,25]
n_f_hat, f_hat = cal_fourier_series_coeff(a, x_boundary, N_f)
Sf = fourier_series_interpolation(x, f_hat, N_f)

# g_tilde
N_g = 2**6
x_g = np.linspace(-pi, pi, N_g)
g_g = np.exp(-a*x_g**2)
n_g_tilde, g_tilde = get_DFT_coeff(g_g, N_g)
Sg = fourier_series_interpolation(x, g_tilde, N_g)

# apply window function for g_tilde
hann = np.hanning(len(g_g))
n_ghann_tilde, ghann_tilde = get_DFT_coeff(g_g*hann, N_g)

# Plot
# G(s) part
plt.figure()
plt.plot(s, G_s, '-k', label='G(s)')
plt.xlabel('s')
plt.ylabel('Function value')
plt.title('a={}'.format(a))
plt.legend()

# f_hat and Sf(x) part
plt.figure()
plt.plot(x, Sf, '-.g', label='Sf(x)')
plt.plot(x, g, ':r', label='g(x)')
plt.xlabel('x')
plt.ylabel('Function value')
plt.title('a={}, N={}'.format(a, N))
plt.legend()

plt.figure()
plt.plot(n_f_hat, f_hat, '-.g', label='f_hat(n)')
plt.bar(n_f_hat, f_hat, align='center')
plt.xlabel('n')
plt.ylabel('Function value')
plt.title('a={}, N={}'.format(a, N_f))
plt.legend()

#g_tilde and Sg(x) part
plt.figure()
plt.plot(n_g_tilde, g_tilde, '-.b', label='g_tilde(n)')
plt.bar(n_g_tilde, g_tilde, align='center')
plt.xlim(-32, 32)
plt.ylim(0, 0.13)
plt.xlabel('n')
plt.ylabel('Function value')
plt.title('a={}, N={}'.format(a, N_g))
plt.legend()

plt.figure()
plt.plot(x, Sg, '-.g', label='Sg(x)')
plt.plot(x, g, ':r', label='g(x)')
plt.xlabel('x')
plt.ylabel('Function value')
plt.title('a={}, N={}'.format(a, N))
plt.legend()

#----Q2(d)----#
N = 2**6
a = 5
x = np.linspace(-pi, pi, N)
g = np.exp(-a*x**2)
n_g, g_tilde = get_DFT_coeff(g, N)

h = np.exp(-2*a*x**2)
n_h, h1_tilde = get_DFT_coeff(h, N)

h2_tilde = np.array(g_tilde)**2

error = np.sqrt((h2_tilde - h1_tilde)**2)

plt.figure()
plt.plot(n_h, h1_tilde, label='h1_tilde')
plt.bar(n_h, h1_tilde, align='center')
plt.xlabel('n')
plt.ylabel('h1_tilde')
plt.title('a={}, N={}'.format(a, N))
plt.legend()

plt.figure()    
plt.plot(n_h, h2_tilde, label='h2_tilde')
plt.bar(n_h, h2_tilde, align='center')
plt.xlabel('n')
plt.ylabel('h2_tilde')
plt.title('a={}, N={}'.format(a, N))
plt.legend()

plt.figure()    
plt.plot(n_h, error)
plt.xlabel('n')
plt.ylabel('error')
plt.title('Error distribution')

plt.show()