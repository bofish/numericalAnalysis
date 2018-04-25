from math import pi, sqrt
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

'''
TODO:
1. Mapping frequency to phisical frequency
2. Mapping Amplitude to phiscal Amplitude
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

def cal_fourier_series(x, a, x_boundary, N):
    # cal f_hat, Sf(x) by analytical way
    x_l = x_boundary[0]
    x_u = x_boundary[1]
    f_fourier_series= [] # Sf(x)
    series_coeff = [] # f_hat
    for x_n in x:
        sum_temp = []
        for n in range(-N//2,N//2):
            integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
            f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi/a)*integ_val/(4*pi)
            sum_temp.append( f_hat_approximation*np.exp(1j*n*x_n) )

            if x_n == x[0]:
                # The coefficents are same for any x
                series_coeff.append(f_hat_approximation)
        f_fourier_series.append(np.sum(sum_temp))
    n_coeff = np.linspace(-N//2, N//2, N)
    return n_coeff, f_fourier_series, series_coeff

def cal_fourier_series_coeff(a, x_boundary, N):
    # cal f_hat by analytical way
    x_l = x_boundary[0]
    x_u = x_boundary[1]
    series_coeff = [] # f_hat
    for n in range(-N//2,N//2):
        integ_val = erf((x_u+n*1j/(2*a))*sqrt(a)) - erf((x_l+n*1j/(2*a))*sqrt(a))
        f_hat_approximation = np.exp(-n**2/(4*a))*sqrt(pi/a)*integ_val/(4*pi)

        series_coeff.append(f_hat_approximation)
    n_coeff = np.linspace(-N//2, N//2, N)
    return n_coeff, series_coeff

def fourier_series_interpolation(x, series_coeff, N):
    # cal Sf(x) by analytical way
    f_fourier_series= [] # Sf(x)
    for x_n in x:
        sum_temp = []
        for n in range(-N//2,N//2):
            sum_temp.append( series_coeff[n+N//2]*np.exp(1j*n*x_n) )
        f_fourier_series.append(np.sum(sum_temp))
    return f_fourier_series

def get_DFT_coeff(g, N):
    g_tilde = abs(FFT(g)/N)
    n = np.linspace(-N//2, N//2+1, N+1)
    g_tilde_shift = FFTshift(g_tilde)
    g_tilde_shift = np.append(g_tilde_shift, 0)
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

#     #----Q2(c)----#
#     N = 2**10
#     a = 5
#     x = np.linspace(-2*pi, 2*pi, N)
#     s = np.linspace(-10, 10, N)
#     g = np.exp(-a*x**2)
    
# # G analytical
#     G_s = cal_fourier_transform(s, a)

# # f_hat
#     x_boundary = [-pi, pi]
#     N_f = 64 # the power of series, i.g. n = [-25,25]
#     n_f_hat, f_hat = cal_fourier_series_coeff(a, x_boundary, N_f)
#     Sf = fourier_series_interpolation(x, f_hat, N_f)

# # g_tilde
#     N_g = 2**6
#     x_g = np.linspace(-pi, pi, N_g)
#     g_g = np.exp(-a*x_g**2)
#     n_g_tilde, g_tilde = get_DFT_coeff(g_g, N_g)
#     Sg = fourier_series_interpolation(x, g_tilde, N_g)

# # apply window function for g_tilde
#     hann = np.hanning(len(g_g))
#     n_ghann_tilde, ghann_tilde = get_DFT_coeff(g_g*hann, N_g)

# # Plot2 
#     a_s = [0.1,  5]
#     for a in a_s:
#         # f_hat
#         n_f_hat, f_hat = cal_fourier_series_coeff(a, x_boundary, N_f)

#         # g_tilde
#         N_g = 2**6
#         x_g = np.linspace(-pi, pi, N_g)
#         g_g = np.exp(-a*x_g**2)
#         n_g_tilde, g_tilde = get_DFT_coeff(g_g, N_g)

#         # apply window function for g_tilde
#         hann = np.hanning(len(g_g))
#         n_ghann_tilde, ghann_tilde = get_DFT_coeff(g_g*hann, N_g)

#         # plt.figure()
#         # plt.plot(n_g_tilde, g_tilde, '-r', label='g_tilde(n)')
#         # plt.bar(n_g_tilde, g_tilde, align='center')
#         # # plt.xlim(-32, 32)
#         # # plt.ylim(0, 0.79)
#         # plt.xlabel('n')
#         # plt.ylabel('Function value')
#         # plt.title('a={}, N={}'.format(a, N_g))
#         # plt.legend()

#         plt.figure()
#         plt.plot(n_g_tilde, ghann_tilde, '-.b', label='ghann_tilde(n)')
#         plt.bar(n_g_tilde, ghann_tilde, align='center')
#         plt.xlim(-32, 32)
#         plt.ylim(0, 0.79)
#         plt.xlabel('n')
#         plt.ylabel('Function value')
#         plt.title('a={}, N={}'.format(a, N_g))
#         plt.legend()

#         # plt.figure()
#         # plt.plot(n_f_hat, f_hat, ':r', label='f_hat(n)')
#         # plt.bar(n_f_hat, f_hat, align='center')
#         # # plt.xlim(-32, 32)
#         # # plt.ylim(0, 0.79)
#         # plt.xlabel('n')
#         # plt.ylabel('Function value')
#         # plt.title('a={}, N={}'.format(a, N_g))
#         # plt.legend()
#     plt.show()
# # Plot
    # # G(s) part
    # plt.figure()
    # plt.plot(s, G_s, '-k', label='G(s)')
    # plt.xlabel('s')
    # plt.ylabel('Function value')
    # plt.title('a={}'.format(a))
    # plt.legend()

    # # f_hat and Sf(x) part
    # plt.figure()
    # plt.plot(x, Sf, '-.g', label='Sf(x)')
    # plt.plot(x, g, ':r', label='g(x)')
    # plt.xlabel('x')
    # plt.ylabel('Function value')
    # plt.title('a={}, N={}'.format(a, N))
    # plt.legend()

    # plt.figure()
    # plt.plot(n_f_hat, f_hat, '-.g', label='f_hat(n)')
    # plt.bar(n_f_hat, f_hat, align='center')
    # plt.xlabel('n')
    # plt.ylabel('Function value')
    # plt.title('a={}, N={}'.format(a, N_f))
    # plt.legend()

    # g_tilde and Sg(x) part
    # plt.figure()
    # plt.plot(n_g_tilde, g_tilde, '-.b', label='g_tilde(n)')
    # plt.bar(n_g_tilde, g_tilde, align='center')
    # # plt.xlim(-32, 32)
    # # plt.ylim(0, 0.13)
    # plt.xlabel('n')
    # plt.ylabel('Function value')
    # plt.title('a={}, N={}'.format(a, N_g))
    # plt.legend()

    # plt.figure()
    # plt.plot(x, Sg, '-.g', label='Sg(x)')
    # plt.plot(x, g, ':r', label='g(x)')
    # plt.xlabel('x')
    # plt.ylabel('Function value')
    # plt.title('a={}, N={}'.format(a, N))
    # plt.legend()

    # plt.show()


    # # f Loop form
    # x_boundary = [-pi, pi]
    # N_f = 64
    # # N_gs = [2, 4, 8, 16, 32, 64, 128]
    # a_s = [0.1, 0.2, 2/pi, 1, 5, 10]
    # a_s = [2/pi]
    # for a in a_s:
    # # N_g = 2**6
    #     n_f_hat, f_hat = cal_fourier_series_coeff(a, x_boundary, N_f)

    #     plt.figure()
    #     plt.plot(n_f_hat, f_hat, '-.g', label='f_hat(n)')
    #     plt.bar(n_f_hat, f_hat, align='center')
    #     plt.xlim(-32, 32)
    #     plt.ylim(0, 0.79)
    #     plt.xlabel('n')
    #     plt.ylabel('Function value')
    #     plt.title('a={}, N={}'.format(a, N_f))
    #     plt.legend(loc=1)

        # g Loop form
    # N_gs = [2, 4, 8, 16, 32, 64, 128]
    # a_s = [0.1, 0.2, 2/pi, 1, 5, 10]
    # a_s = [2/pi]
    # for a in a_s:
    # # N_g = 2**6
    #     x_g = np.linspace(-pi, pi, N_g)
    #     g = np.exp(-a*x_g**2)
    #     n_g_tilde, g_tilde = get_DFT_coeff(g, N_g)

    #     plt.figure()
    #     plt.plot(n_g_tilde, g_tilde, '-.g', label='g_tilde(n)')
    #     plt.bar(n_g_tilde, g_tilde, align='center')
    #     plt.xlim(-32, 32)
    #     plt.ylim(0, 0.79)
    #     plt.xlabel('n')
    #     plt.ylabel('Function value')
    #     plt.title('a={}, N={}'.format('2/pi', N_g))
    #     plt.legend()

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
    # plt.plot(n_g, g_tilde)
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