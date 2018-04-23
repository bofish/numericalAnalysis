import numpy as np
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
import numpy as np
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    # print(M)
    # print(k * n)
    return np.dot(M, x)
# x = np.random.random(1024)
# x = np.arange(10)
# np.allclose(DFT_slow(x), np.fft.fft(x))
# DFT_slow(x)
# a = [1.0 + 2.0j, 1.0 + 2.0j, ]
# b = [2.0, 4.0]
# print(np.dot(a, b))

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    print(N)
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:(N / 2)] * X_odd,
                               X_even + factor[(N / 2):] * X_odd])
x = np.random.random(1024)
# print(np.fft.fft(x))
print(np.allclose(FFT(x), np.fft.fft(x)))
