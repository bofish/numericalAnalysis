import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fourier import convolution_spectral, RK4_2d, FFT2, IFFT2

def get_alpha_beta(N):
    alpha = np.array([np.arange(-N/2, N/2) for i in range(N)])
    beta = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    return (alpha, beta)

def get_U_V(W, alpha, beta):
    ab = alpha**2 + beta**2
    ab[N//2, N//2] = 1
    U = 1j*beta*W/ab
    V = -1j*alpha*W/ab
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
    w = 2/R*(1 - r_bar**2/2)*np.exp(0.5*(1 - r_bar**2))
    W = init_process(np.fft.fft2(w))
    return w, W

def taylor_vortex(N, L, dt, Nt, Re):
    t0 = 0
    w0, W0 = taylor_init(N, L)
    vt = np.zeros(Nt)
    vw = np.array([np.zeros((N, N)) for i in range(Nt)])
    
    # Initial condition
    vt[0] = t = t0
    vw[0] = w = w0

    # Create r_jk matrix 
    j = np.array([np.arange(-N/2, N/2) for i in range(N)])
    k = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    r = 2*np.pi/N*np.sqrt(j**2 + k**2)
    R = L/np.sqrt(2)
    r_bar = r/R
    for i in range(1, Nt):
        vt[i] = t = t0 + i * dt
        t_bar = 2*t/(Re*R**2)
        vw[i] = w = 2/(R*(1 + t_bar)**2)*(1 - r_bar**2/(2 + 2*t_bar))*np.exp(0.5*(1 - r_bar**2/(1 + t_bar)))
    return vt, vw

if __name__ == '__main__':
    L = 1.5
    N = 64
    t0 = 0.0
    t1 = 1
    Nt = 1000
    dt = (t1 - t0)/(Nt - 1)
    Re = 100000

    (w0, W0) = taylor_init(N, L)
    # (vt, vW) = RK4_2d(get_dW, W0, N, dt, Nt)
    
    # vw = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    # for i in range(Nt):
    #     vw[i] = np.fft.ifft2(vW[i])
    w0_2 = np.fft.ifft2(W0)
    print(np.allclose(w0_2, w0))
    plt.figure()
    ax = sns.heatmap(w0_2.real, cmap="YlGnBu") 
    plt.figure()
    ax = sns.heatmap(w0.real, cmap="YlGnBu") 

    # 
    # plt.figure(1)
    # p = int(321)
    # for i in np.linspace(1, Nt, 6):
    #     i = int(i)-1
    #     print(i)
    #     plt.subplot(p)
    #     ax = sns.heatmap(vw[i].real, cmap="YlGnBu") 
    #     plt.title(i) 
    #     p += 1

    plt.show()
    

