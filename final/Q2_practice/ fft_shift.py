import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 64

# Forward fourier transform
x = np.random.randn(N, N)
x_ishift = np.fft.ifftshift(x)
X = np.fft.fft2(x_ishift)
x2 = np.fft.fftshift(np.fft.ifft2(X))
print(np.allclose(x, x2))

# # Backward fourier transform
# X = np.random.randn(N, N) + 1j*np.random.randn(N, N)
# x = np.fft.ifft2(np.fft.ifftshift(X))
# X2 = np.fft.fftshift(np.fft.fft2(x))
# print(np.allclose(X, X2))
