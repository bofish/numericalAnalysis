from math import pi, sqrt
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=np.complex_)
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
    x = np.asarray(x, dtype=np.complex_)
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
    """A recursive implementation of the 1D Cooley-Tukey IFFT"""
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

def DFT2(x):
    """Compute the discrete Fourier Transform of the 2D array x"""
    x = np.asarray(x, dtype=np.complex_)
    N = x.shape[0]
    X = np.zeros((N, N), dtype=np.complex_)
    for alpha in range(N):
        for beta in range(N):
            sumj = []
            for j in range(N):
                sumk = []
                for k in range(N):
                    sumk.append(x[j,k] * np.exp(-1j*2*np.pi/N*(alpha*j+beta*k)))
                sumj.append(np.sum(sumk))
            X[alpha,beta] = np.sum(sumj)
    return X

def IDFT2(X):
    """Compute the discrete Fourier Transform of the 2D array x"""
    X = np.asarray(X, dtype=np.complex_)
    N = X.shape[0]
    x = np.zeros((N, N), dtype=np.complex_)
    for j in range(N):
        for k in range(N):
            suma = []
            for alpha in range(N):
                sumb = []
                for beta in range(N):
                    sumb.append(X[alpha, beta] * np.exp(1j*2*np.pi/N*(alpha*j+beta*k)))
                suma.append(np.sum(sumb))
            x[j,k] = np.sum(suma)
    return x/N**2

def FFT2(x):
    """A recursive implementation of the 2D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=np.complex_)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT2(x)
    else:
        X_row = np.array([FFT(x[i,:]) for i in range(N)])
        X_column = np.concatenate([np.array([FFT(X_row[:,i])]).T for i in range(N)], axis=1)
        return X_column

def IFFT2(X):
    """A recursive implementation of the 2D Cooley-Tukey FFT"""
    X = np.asarray(X, dtype=np.complex_)
    N = X.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return IDFT2(X)
    else:
        x_row = np.array([IFFT(X[i,:]) for i in range(N)])
        x_column = np.concatenate([np.array([IFFT(x_row[:,i])]).T for i in range(N)], axis=1)
        return x_column
def de_non_periodicity(x):
    N = x.shape[0]
    h = np.hanning(N)   
    ham2d = np.sqrt(np.outer(h,h))
    return x*ham2d

def zero_padding(X):
    """A zero-padding technique with the 3/2 de-aliasing rule"""
    X = np.asarray(X, dtype=np.complex_)
    N = X.shape[0]
    M = int(3/2*N)
    X_pad = np.zeros((M, M), dtype=np.complex_)
    lb = (M - N)//2
    ub = (M + N)//2
    X_pad[lb:ub, lb:ub] = X
    x_pad = IFFT2(X_pad)
    return x_pad

def convolution_spectral(U, V):
    """Apply spectral method to take convolution """
    # Step2&3: zero padding and IFFT
    u_pad = zero_padding(U)
    v_pad = zero_padding(V)
    
    # Step4: Perform the multiplication w = u*.v
    w_pad = u_pad*v_pad

    # Step5: FFT
    W_pad = FFT2(w_pad)
   
    # Step6: Chop off
    M = u_pad.shape[0]
    N = U.shape[0]
    lb = (M - N)//2
    ub = (M + N)//2
    W = W_pad[lb:ub, lb:ub]
    return W

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

def RK4_2d(f, y0, N, dt, t1):
    t0 = 0
    Nt = (t1 - t0)/dt + 1
    
    vt = np.zeros(Nt)
    vy = np.array([np.zeros((N, N)) for i in range(Nt)])
    vt[0] = t = t0
    vy[0] = y = y0

    for i in range(1, Nt):
        k1 = dt * f(y)
        k2 = dt * f(y + 0.5*k1)
        k3 = dt * f(y + 0.5*k2)
        k4 = dt * f(y + k3)
        vt[i] = t = t0 + i * dt
        vy[i] = y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return vt, vy

def stability_monitor(x):
    dt_max = np.max([1,2])
    return dt_max

if __name__ == '__main__':
    # x = np.random.random(2**10)
    # print(np.allclose(IFFT(x), np.fft.ifft(x)))

    x = np.random.randn(16,16)
    # print(np.allclose(IFFT2(x), np.fft.ifft2(x)))
    # U = np.random.randn(8,8)
    # W = np.random.randn(8,8)
    # UW = convolution_spectral(U,W)
    # print(UW.shape)
    # VW = convolution_spectral(V,W)

    x = de_non_periodicity(x)
    print(x)

    t = np.arange(16)


