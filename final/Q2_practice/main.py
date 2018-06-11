import numpy as np
import matplotlib.pyplot as plt
from fourier import convolution_spectral, RK4_2d

def get_U_V(W, alpha, beta):
    if alpha == 0 and beta == 0:
        U = 0
        V = 0
    else:
        U = 1j*beta*W/(alpha**2 + beta**2)
        V = -1j*alpha*W/(alpha**2 + beta**2)
    return (U, V)

def get_dW(W, alpha, beta, Re):
    (U, V) = get_U_V(W, alpha, beta)
    UW = convolution_spectral(U, W)
    VW = convolution_spectral(V, W)
    dW = -1j*alpha*UW - 1j*beta*VW - (alpha**2 + beta**2)*W/Re
    return dW

if __name__ == '__main__':
    RK4_2d(get_dW, )


