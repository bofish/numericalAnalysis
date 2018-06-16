from math import pi, sqrt
import numpy as np
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
    x_pad = np.fft.ifft2(X_pad)
    return x_pad

def convolution_spectral(U, V):
    """Apply spectral method to take convolution """
    # Step2&3: zero padding and IFFT
    u_pad = zero_padding(U)
    v_pad = zero_padding(V)
    
    # Step4: Perform the multiplication w = u*.v
    w_pad = u_pad*v_pad

    # Step5: FFT
    W_pad = np.fft.fft2(w_pad)
   
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

if __name__ == '__main__':
    import seaborn as sns 
    N = 256
    x = np.random.randn(N,N)
    print(np.allclose(FFT2(x), np.fft.fft2(x)))

    x_hann = de_non_periodicity(x)
    X_hann = np.fft.fft2(x_hann)
    X_ori = np.fft.fft2(x)

    plt.figure()
    ax = sns.heatmap(X_hann.real, cmap="YlGnBu") 
    plt.figure()
    ax = sns.heatmap(X_ori.real, cmap="YlGnBu") 
    plt.show()


