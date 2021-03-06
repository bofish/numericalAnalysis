import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_alpha_beta(N):
    alpha = np.array([np.arange(-N/2, N/2) for i in range(N)])
    beta = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    return (alpha, beta)

def get_U_V(W):
    N = W.shape[0]
    (alpha, beta) = get_alpha_beta(N)
    ab = alpha**2 + beta**2
    ab[N//2, N//2] = 1
    U = 1j*beta*W/ab
    V = -1j*alpha*W/ab
    U[N//2, N//2] = 0 + 0j
    V[N//2, N//2] = 0 + 0j
    return (U, V)

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

def get_dW(W, Re):
    # Re = 1000
    N = W.shape[0] 
    (alpha, beta) = get_alpha_beta(N)
    (U, V) = get_U_V(W)
    UW = convolution_spectral(U, W)
    VW = convolution_spectral(V, W)
    dW = -1j*alpha*UW - 1j*beta*VW - (alpha**2 + beta**2)*W/Re
    return dW

def RK4_2d(dW, W0, N, dt, Nt, Re):
    t0 = 0
    vt = np.zeros(Nt)
    vW = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    vU = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])
    vV = np.array([np.zeros((N, N), dtype=np.complex_) for i in range(Nt)])

    vt[0] = t = t0
    vW[0] = W = W0
    (vU[0], vV[0]) = (U, V) =get_U_V(W0)
    
    for i in range(1, Nt):
        # print(i)
        k1 = dt * dW(W, Re)
        k2 = dt * dW(W + 0.5*k1, Re)
        k3 = dt * dW(W + 0.5*k2, Re)
        k4 = dt * dW(W + k3, Re)
        vt[i] = t = t0 + i * dt
        vW[i] = W = W + (k1 + 2*k2 + 2*k3 + k4) / 6
        (vU[i], vV[i]) = (U, V) =get_U_V(W)
    dt_max = monitor_time_stability(vU, vV, dt, Re, N)
    # dt_max = 0
    return vt, vW, dt_max

def monitor_time_stability(vU, vV, dt, Re, N):
    vu = np.fft.ifft2(vU)
    vv = np.fft.ifft2(vV)

    dx = 2*np.pi/N
    u_max = np.max(vu)
    v_max = np.max(vv)
    V_max = np.max([u_max, v_max])
    dt_max = np.min([2.82*dx/(np.pi*V_max), 2.8*Re*dx**2/np.pi**2])
    print('V_max: {}'.format(V_max))
    if dt > dt_max:
        print('Time step over the limit: {}'.format(dt - dt_max))
        raise Exception('Times stability leakage')        
    return dt_max

def taylor_init(N, L, R='Auto', init=False):
    if R == 'Auto':
        R = L/np.sqrt(2)
    j = np.array([np.arange(-N/2, N/2) for i in range(N)])
    k = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    r = 2*np.pi/N*np.sqrt(j**2 + k**2)
    r_bar = r/R
    w = 2/R*(1 - r_bar**2/2)*np.exp(0.5*(1 - r_bar**2))
    u = r_bar*np.exp(0.5*(1 - r_bar**2))
    if init:
        W = init_process(np.fft.fft2(w))
    else:
        W = np.fft.fft2(w)
    return w, W, u

def init_process(W0):
    N = W0.shape[0]
    W0_processed = np.copy(W0)
    W0_processed[N//2, N//2] = 0
    W0_processed[:,0] = 0
    W0_processed[0,:] = 0
    return W0_processed

def taylor_vortex(N, L, dt, Nt, Re, R='Auto'):
    t0 = 0
    w0, W0, u0 = taylor_init(N, L, R=R, init=False)
    vt = np.zeros(Nt)
    vw = np.array([np.zeros((N, N)) for i in range(Nt)])
    vu = np.array([np.zeros((N, N)) for i in range(Nt)])

    # Initial condition
    vt[0] = t = t0
    vw[0] = w = w0
    vu[0] = u = u0

    # Create r_jk matrix 
    j = np.array([np.arange(-N/2, N/2) for i in range(N)])
    k = np.array([np.arange(-N/2, N/2) for i in range(N)]).T
    r = 2*np.pi/N*np.sqrt(j**2 + k**2)
    if R == 'Auto':
        R = L/np.sqrt(2)
    r_bar = r/R
    t_init = 0.5*Re*R**2
    for i in range(1, Nt):
        vt[i] = t = t0 + i * dt
        t_bar = 2*t/(R**2)
        vw[i] = w = 2/(R*(1 + t_bar)**2)*(1 - r_bar**2/(2 + 2*t_bar))*np.exp(0.5*(1 - r_bar**2/(1 + t_bar)))
        vu[i] = u = r_bar/(1 + t_bar)**2*np.exp(0.5*(1 - r_bar**2/(1 + t_bar)))
    return vt, vw, vu

def show_heat_map(data2D, time_lbl, vmin=-3.0, vmax=3.0):
    N_data = data2D.shape[0]
    fg = plt.figure()
    plt.tight_layout()
    p = int(221)
    for i in np.linspace(1, N_data, 4):
        i = int(i)-1
        print(i)
        plt.subplot(p)
        plt.tight_layout()
        # ax = sns.heatmap(data2D[i].real, cmap="RdBu_r") 
        ax = sns.heatmap(data2D[i].real, robust=True, fmt="f", cmap='RdBu_r', vmin=vmin, vmax=vmax) 
        plt.title('t={0:4.2f}'.format(time_lbl[i]))
        p += 1
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.suptitle('b={}'.format(b))
    return fg

def RMS2(mat):
    N = mat.shape[0]
    return np.sqrt(np.sum(mat**2)/N**2)

if __name__ == '__main__':
    L = 1.5
    N = 64
    t0 = 0.0
    t1 = 0.1
    Nt = 100
    dt = (t1 - t0)/(Nt - 1)
    Re = 1000

    # Numerical solution
    (w0, W0, u0) = taylor_init(N, L, init=False)
    (vt, vW, dt_max) = RK4_2d(get_dW, W0, N, dt, Nt, Re)
    
    vw = np.fft.ifft2(np.fft.ifftshift(vW))
   
    fg1 = show_heat_map(vw, vt)
    plt.show()


    # w0_2 = np.fft.ifft2(W0)
    # print(np.allclose(w0_2, w0))
    # plt.figure()
    # ax = sns.heatmap(w0_2.real, cmap="YlGnBu") 
    # plt.figure()
    # ax = sns.heatmap(w0.real, cmap="YlGnBu") 
    

