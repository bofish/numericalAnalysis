import numerical_taylor as nt
import exact_taylor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':
    L = 1.5
    R = 'Auto'
    t0 = 0.0
    t1 = 0.001
    Nt = 100
    dt = (t1 - t0)/(Nt - 1)
    Re = 1000
    N_f = 256
    Ns = [256, 128, 64, 32]
    for N in Ns:
        (w0, W0, u0) = nt.taylor_init(N, L, R, init=False)
        (t, W, dt_max) = nt.RK4_2d(nt.get_dW, W0, N, dt, Nt, Re)
        w = np.fft.ifft2(W)
        if N == Ns[0]:
            w_f = w
        elif N == Ns[1]:
            dN = Ns[0]//N
            w_c1 = w
            w_f1 = w_f[:, ::dN, ::dN]
        elif N == Ns[2]:
            dN = Ns[0]//N
            w_c2 = w
            w_f2 = w_f[:, ::dN, ::dN]
        elif N == Ns[3]:
            dN = Ns[0]//N
            w_c3 = w
            w_f3 = w_f[:, ::dN, ::dN]
    
    # Error estimate
    errs1 = nt.RMS2(w_f1[-1] - w_c1[-1])
    errs2 = nt.RMS2(w_f2[-1] - w_c2[-1])
    errs3 = nt.RMS2(w_f3[-1] - w_c3[-1])
    errs = [errs1, errs2, errs3]

    print('error 128: {}, 64: {}, 32:{}'.format(errs1, errs2, errs3))

    plt.figure()
    plt.semilogy(Ns[1:], errs, label='m={:4.2f}'.format(1))
    plt.legend()
    plt.title('Finite difference with convergence rate')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.grid()

    # Heat map
    fg1 = nt.show_heat_map(w_f, t)
    fg2 = nt.show_heat_map(w_c1, t)
    fg3 = nt.show_heat_map(w_c2, t)
    fg4 = nt.show_heat_map(w_c3, t)
    plt.show()