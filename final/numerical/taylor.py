import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numerical import ode
from numerical import fourier

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

def get_dW(W, Re):
    # Re = 1000
    N = W.shape[0] 
    (alpha, beta) = get_alpha_beta(N)
    (U, V) = get_U_V(W)
    UW = fourier.convolution_spectral(U, W)
    VW = fourier.convolution_spectral(V, W)
    dW = -1j*alpha*UW - 1j*beta*VW - (alpha**2 + beta**2)*W/Re
    return dW

def monitor_time_stability(vU, vV, dt, Re, N):
    vu = fourier.IFFT2(vU)
    vv = fourier.IFFT2(vV)

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
        W = init_process(fourier.FFT2(w))
    else:
        W = fourier.FFT2(w)
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
        plt.suptitle('Heat map')
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
    (vt, vW, dt_max) = ode.RK4_2d(get_dW, W0, N, dt, Nt, Re)
    vw = fourier.IFFT2(np.fft.ifftshift(vW))
   
    fg1 = show_heat_map(vw, vt)
    plt.show()
