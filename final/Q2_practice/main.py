import numpy as np
import matplotlib.pyplot as plt
from fourier import convolution_spectral, RK4_2d, FFT2, IFFT2

def get_alpha_beta(N):
    alpha = np.array([np.arange(-N/2, N/2) for i in range(N)])
    beta = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    return (alpha, beta)

def get_U_V(W, alpha, beta):
    alpha[N//2, N//2] = 1
    beta[N//2, N//2] = 1
    U = 1j*beta*W/(alpha**2 + beta**2)
    V = -1j*alpha*W/(alpha**2 + beta**2)
    U[N//2, N//2] = 0 + 0j
    V[N//2, N//2] = 0 + 0j
    return (U, V)

def get_dW(W):
    Re = 1000
    N = W.shape[0] 
    (alpha, beta) = get_alpha_beta(N)
    (U, V) = get_U_V(W, alpha, beta)
    UW = convolution_spectral(U, W)
    VW = convolution_spectral(V, W)
    dW = -1j*alpha*UW - 1j*beta*VW - (alpha**2 + beta**2)*W/Re
    return dW

def taylor_init(N, L):
    R = L/np.sqrt(2)
    j = np.array([np.arange(-N/2, N/2) for i in range(N)])
    k = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    r = 2*np.pi/N*np.sqrt(j**2 + k**2)
    r_bar = r/R
    w = 2/R*(1 - r_bar**2/2)*np.exp(0.5*(1-r_bar**2))
    W = init_process(np.fft.fft2(w))
    return W

def init_process(W0):
    N = W0.shape[0]
    W0_processed = np.copy(W0)
    W0_processed[N//2, N//2] = 0
    W0_processed[:,0] = 0
    W0_processed[0,:] = 0
    return W0_processed

if __name__ == '__main__':
    L = 1.5
    N = 128
    t0 = 0.0
    t1 = 3
    Nt = 700
    dt = (t1 - t0)/(Nt - 1)
    
    W0 = taylor_init(N, L)
    (vt, vW) = RK4_2d(get_dW, W0, N, dt, Nt)
    print(vW)
    print(vt)
    
    vw = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    for i in range(Nt):
        print('IFFT {}'.format(i))
        vw[i] = np.fft.ifft2(vW[i])
    print(vw)

    import seaborn as sns
    plt.figure(1)
    p = int(321)
    for i in np.linspace(1, Nt, 6):
        i = int(i)-1
        print(i)
        plt.subplot(p)
        ax = sns.heatmap(vw[i].real, cmap="YlGnBu") 
        plt.title(i) 
        p += 1
    plt.show()

