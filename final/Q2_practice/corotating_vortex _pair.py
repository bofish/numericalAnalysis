import numerical_taylor as nt
import exact_taylor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def init_condition(N, n, b, L):
    c = 1.5
    c1 = L*c/N
    d = L*b/N
    print(c1,d)
    ct1 = N//2 + b//2
    ct2 = N//2 - b//2
    w0_single, W0, u0 = nt.taylor_init(n, c, R='Auto', init=False)
    
    w_init = np.zeros((N, N))
    w_init[N//2, N//2] = 1
    w_init[N//2 - n//2:N//2 + n//2, ct1 - n//2:ct1 + n//2] += w0_single
    w_init[N//2 - n//2:N//2 + n//2, ct2 - n//2:ct2 + n//2] += w0_single
    w_init[N//2, ct2] = 1
    w_init[N//2, ct1] = 1
    return w_init

def co_taylor_vortex(N, n, b, L, dt, Nt, Re):
    c = 1.0
    d = L*b/N
    ct1 = N//2 + b//2
    ct2 = N//2 - b//2
    vt, w_single, u = nt.taylor_vortex(n, c, dt, Nt, Re, R='Auto')
    vw = np.array([np.zeros((N, N)) for i in range(Nt)])
    vw[:, N//2, N//2] = 0
    vw[:, N//2 - n//2:N//2 + n//2, ct1 - n//2:ct1 + n//2] += w_single
    vw[:, N//2 - n//2:N//2 + n//2, ct2 - n//2:ct2 + n//2] += w_single

    return vt, vw


if __name__ == '__main__': 
    L = 1
    R = 'Auto'
    
    t0 = 0.0
    t1 = 0.5
    Nt = 500
    dt = (t1 - t0)/(Nt - 1)
    Re = 100
    N = 256
    n = 128
    # b = 25
    bs = [8, 16, 32, 64, 128]
    ln = len(bs)
    wlim = 8.627880029445057

    for i in range(ln):
        b = bs[i]
        t, w = co_taylor_vortex(N, n, b, L, dt, Nt, Re)
        # wlim = np.max(w)
        nt.show_heat_map(w, t,b, vmin=-wlim, vmax=wlim)

    plt.show()