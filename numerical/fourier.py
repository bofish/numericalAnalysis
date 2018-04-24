from math import pi, sqrt
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

def cal_fourier_transform(s, a):
    # cal G(s) by analytical way
    return sqrt(pi/a)*np.exp(-pi**2*s**2/a)

def cal_fourier_series(x, a, x_boundary):
    # cal f_hat, Sf(x) by analytical way
    x_l = x_boundary[0]
    x_u = x_boundary[1]
    f_fourier_series= [] # Sf(x)
    series_coeff = [] # f_hat
    for x_n in x:
        sum_temp = []
        for n in range(-50,50):
            integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
            f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi)*integ_val/(4*pi*sqrt(a))
            sum_temp.append( f_hat_approximation*np.exp(1j*n*x_n) )

            if x_n == x[0]:
                # The coefficents are same for any x
                series_coeff.append(f_hat_approximation)
        f_fourier_series.append(np.sum(sum_temp))
    n_coeff = np.linspace(-10, 10, len(series_coeff))
    return n_coeff, f_fourier_series, series_coeff

def get_DFT_coeff(g, N):
    g_tilde = abs(FFT(g)/N)
    n = np.linspace(-10, 10, N)
    g_tilde_shift = FFTshift(g_tilde)

    return n, g_tilde_shift

if __name__ == '__main__':
    # x = np.random.random(2**10)
    # print(np.allclose(IFFT(x), np.fft.ifft(x)))
    # X = DFT_slow(x)
    # x2 = IDFT_slow(X)


    # print(np.allclose(x, x2))
    # print(np.allclose(IDFT_slow(X), np.fft.ifft(X)))
    # #----Q2(a)----#
    # N = 2**10
    # a = 10
    # # G analytical
    # s = np.linspace(-10, 10, N)
    # G_analytical = sqrt(pi/a)*np.exp(-pi**2*s**2/a)
    # plt.figure()
    # plt.plot(s, G_analytical)
    # plt.xlabel('s')
    # plt.ylabel('G(s) (Analytical Expression)')
    # plt.title('a={}'.format(a))

    # # f function analytical
    # x = np.linspace(-10*pi, 10*pi, N)
    # g = np.exp(-a*x**2)
    # f_analytical = []
    # for x_m in x:
    #     f_m = []
    #     for m in np.arange(-1,2):
    #         f_m.append(np.exp(-a*(x_m-2*pi*m)**2))
    #     f_analytical.append(np.sum(f_m))

    # # f_hat analytical
    # x_u = 2*pi
    # x_l = -2*pi
    # f_fourier_analytical = []
    # f_hat = []
    # for x_n in x:
    #     sum_temp = []
    #     for n in range(-25, 25):
    #         integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
    #         f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi)*integ_val/(4*pi*sqrt(a))
    #         sum_temp.append( f_hat_approximation*np.exp(1j*n*x_n) )

    #         if x_n == x[0]:
    #             # The coefficents are same for any x
    #             f_hat.append(f_hat_approximation)

    #     f_fourier_analytical.append(np.sum(sum_temp))
    # print(np.sum(f_hat))
    # plt.figure()
    # # plt.plot(x, g, '-k',label='g(x)')
    # plt.plot(x, f_analytical, '-.r', label='f(x) (Analytical Expression)')
    # # plt.plot(x, np.real(f_fourier_analytical), '-g', label='Sf(x) (Truncated Foureir series)')
    # plt.xlabel('x')
    # plt.ylabel('function value')
    # plt.title('a={}'.format(a))

    # plt.figure()
    # print(len(f_hat))
    # plt.plot(f_hat)
    # plt.xlabel('n')
    # plt.ylabel('f_hat(a)')
    # plt.title('a={}'.format(a))



    # x = np.random.random(2**10)
    # print(np.allclose(FFT(g)/1024, IFFT(g)))
    # X = DFT_slow(x)
    # x2 = IDFT_slow(X)

    # # ----Q2(b)----#
    # N = 2**7
    # a = 5
    # x = np.linspace(-pi, pi, N)
    # g = np.exp(-a*x**2)
    # g_tilde = FFT(g)/N
    # n = np.linspace(-N/2, N/2, N)
    # g_tilde_shift = FFTshift(g_tilde)
    
    # a_s = [2/pi, 1, 5, 10]
    # ls = ['-', ':', '-.', '--']
    # plt.figure(7)
    # for index in range(4):
    #     a = a_s[index]
    #     g = np.exp(-a*x**2)
    #     g_tilde = FFT(g)/N
    #     n = np.linspace(-50, 50, N)
    #     g_tilde_shift = FFTshift(g_tilde)
    #     plt.plot(n, abs(g_tilde_shift), linestyle=ls[index])


    # # plt.figure(7)
    # # plt.plot(n, abs(g_tilde_shift), label='g_tilde(a)')
    # plt.xlabel('n')
    # plt.ylabel('g_tilde(a)')
    # plt.legend(['a=2/pi', 'a=1', 'a=5', 'a=10'], loc=2)
    # plt.title('N={}'.format(N))

    # f_fourier = []  # Fourier series interpolation
    # for x_n in x:
    #     single_term = []
    #     for n in range(-N//2, N//2):
    #         single_term.append(g_tilde[n]*np.exp(1j*n*x_n))
    #     f_fourier.append(np.sum(single_term))
    # # print(f)
    # f_fourier = FFTshift(f_fourier)
    # plt.figure()
    # plt.plot(g, '-k', label='g(x)')
    # plt.plot(np.real(f_fourier), ':g', label='Sf(x)')
    # plt.xlabel('x')
    # plt.ylabel('function value')
    # plt.title('a={}, N={}'.format(a, N))

    #----Q2(c)----#
    N = 2**10
    a = 5
    # G analytical
    s = np.linspace(-10, 10, N)
    G_s = cal_fourier_transform(s, a)

    # f_hat
    x = np.linspace(-pi, pi, N)
    x_boundary = [-pi, pi]
    n_f_hat, Sf, f_hat = cal_fourier_series(x, a, x_boundary)
    print(len(f_hat))

    # g_tilde
    g = np.exp(-a*x**2)
    n_g_tilde, g_tilde = get_DFT_coeff(g, N)

    plt.figure()
    plt.plot(s, G_s, ':r', label='G(s)')
    plt.xlabel('s')
    plt.figure()
    plt.plot(n_f_hat, f_hat, '-.g', label='f_hat(n)')
    plt.xlabel('n')
    plt.figure()
    plt.plot(n_g_tilde, g_tilde, '--b', label='g_tilde(n)')
    plt.xlabel('n')
        
    plt.legend(loc=2)
    plt.show()

