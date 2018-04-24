from math import pi, sqrt, inf
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

'''
TODO:
1. Wirriten down own fft algorithm
2. Veirfy fourier transfrom by `np.fft.fft()` and analytical method
3. Mapping frequency to phisical frequency
4. Mapping Amplitude to phiscal Amplitude
'''

# # Example 1, pure sin and cos function as input

# # Parameters
# fs = 1000.0 # Sampling rate in Hz, must larger than Sc/2
# dt = 1.0/fs # Sample time in sec
# Sc1 = 50.0 # Fucntion frequency
# Sc2 = 80.0 # Fucntion frequency
# x = np.linspace(0.0, 1000*dt, 1000)
# y = np.sin(Sc1 * 2.0*np.pi*x) + 0.5*np.sin(Sc2 * 2.0*np.pi*x)

# # Numerical FFT by numpy library 
# Y = np.fft.fft(y)
# N = len(Y) # half the length of the output of the FFT.
# X = np.linspace(-fs/2, fs/2, N) # Map frequency to real frequency by Nyquist frequency theorem

# # Applied windows function
# hann = np.hanning(len(y))
# Yhann = np.fft.fft(hann*y)
# plt.plot(X, 4*np.abs(Yhann[:N])/N)
# plt.grid()
# plt.show()

# # Example 2, Guassian fucntion as input

# # Parameters
# a = 10
# x = np.linspace(-10, 10, 2000)
# g = np.exp(-a*x**2) 

# # Numerical FFT by numpy library 
# G = np.fft.fft(g)
# N = len(Y)//2 # half the length of the output of the FFT.
# dt = x[1] - x[0]
# fs = 1/dt
# X = np.linspace(0, fs/2, N) # Map frequency to real frequency by Nyquist frequency theorem

# # Applied windows function
# hann = np.hanning(len(g))
# Ghann = np.fft.fft(hann*g)
# plt.plot(X, 2.0*np.abs(Ghann[:N])/N)

# # Analytical fourier transform 
# AnalG = np.sqrt(np.pi/a)*np.exp(-np.pi**2*X**2/a)
# plt.plot(X, AnalG)


# # Analytical fouier transform
# # G_analytical = sqrt(pi)*np.exp(-pi**2*s**2/a**2)

# plt.show()

# Follow the tutorial

# def DFT_slow(x):
#     """Compute the discrete Fourier Transform of the 1D array x"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(-2j * np.pi * k * n / N)
#     # print(M)
#     # print(k * n)
#     return np.dot(M, x)
# # x = np.random.random(1024)
# # x = np.arange(10)
# # np.allclose(DFT_slow(x), np.fft.fft(x))
# # DFT_slow(x)
# # a = [1.0 + 2.0j, 1.0 + 2.0j, ]
# # b = [2.0, 4.0]
# # print(np.dot(a, b))

# def FFT(x):
#     """A recursive implementation of the 1D Cooley-Tukey FFT"""
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     print(N)
#     if N % 2 > 0:
#         raise ValueError("size of x must be a power of 2")
#     elif N <= 32:  # this cutoff should be optimized
#         return DFT_slow(x)
#     else:
#         X_even = FFT(x[::2])
#         X_odd = FFT(x[1::2])
#         factor = np.exp(-2j * np.pi * np.arange(N) / N)
#         return np.concatenate([X_even + factor[:(N / 2)] * X_odd,
#                                X_even + factor[(N / 2):] * X_odd])
# x = np.random.random(1024)
# # print(np.fft.fft(x))
# print(np.allclose(FFT(x), np.fft.fft(x)))

# IDFT

def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)    
    return np.dot(M, x)

def IDFT(X):
    """Compute the discrete Fourier Transform of the 1D array x"""
    X = np.asarray(X)
    N = X.shape[0]
    k = np.arange(N)
    n = k.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X)/N

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:(N // 2)] * X_odd,
                               X_even + factor[(N // 2):] * X_odd])

def IFFT(X):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    X = np.asarray(X)
    N = X.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return IDFT(X)
    else:
        x_even = IFFT(X[::2])*(N/2)
        x_odd = IFFT(X[1::2])*(N/2)
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([x_even + factor[:(N // 2)] * x_odd,x_even + factor[(N // 2):] * x_odd])/N

def FFTshift(x):
    y = np.asarray(x)
    n = y.shape[0]
    p2 = (n+1)//2
    shiftlist = np.concatenate((np.arange(p2, n), np.arange(p2)))
    y = np.take(y, shiftlist)
    return y



if __name__ == '__main__':
    # x = np.random.random(2**10)
    # print(np.allclose(IFFT(x), np.fft.ifft(x)))
    # X = DFT_slow(x)
    # x2 = IDFT_slow(X)


    # print(np.allclose(x, x2))
    # print(np.allclose(IDFT_slow(X), np.fft.ifft(X)))
    #----Q2(a)----#
    N = 2**10
    a = 2/pi

    # G analytical
    s = np.linspace(-5, 5, N)
    G_analytical = sqrt(pi)*np.exp(-pi**2*s**2/a**2)
    plt.figure()
    plt.plot(s, G_analytical, label='G(s) (Analytical Expression)')

    # f function analytical
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

    #----Q2(b)----#
    x = np.linspace(-pi, pi, N)
    g_tilde = IFFT(g)
    print(g_tilde.shape[0])

    f_fourier = []  # Fourier series interpolation
    for x_n in x:
        single_term = []
        for n in range(-N//2, N//2):
            single_term.append(g_tilde[n]*np.exp(1j*n*x_n))
        f_fourier.append(np.sum(single_term))
    # print(f)
    f_fourier = FFTshift(f_fourier)
    plt.figure()
    plt.plot(np.real(f_fourier))
    plt.plot(g)
    plt.show()

