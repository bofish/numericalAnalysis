import numerical.fourier as nf
import numerical.taylor as nt
import numerical.ode as node
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    #----- Q2(b)(i) stability bound test -----#
    L = 1.5
    N = 16
    t0 = 0.0
    t1 = 1.0
    Nt = 150
    dt = (t1 - t0)/(Nt - 1)
    
    W0_rand = nt.init_process(np.random.randn(N, N) + 1j*np.random.randn(N, N))
    vdt_max = []
    vRe = range(10,500,10)
    for Re in vRe:
        print(Re)
        (vt, vW, dt_max) = node.RK4_2d(nt.get_dW, W0_rand, N, dt, Nt, Re)
        vdt_max.append(dt_max)
    plt.plot(vRe, np.real(vdt_max))
    plt.title('The stability bounds of the time step')
    plt.xlabel('Re')
    plt.ylabel('dt_max')
    
    #----- Q2(b)(ii) time convergence test -----#
    L = 1.5
    R = 'Auto'
    N = 64
    t0 = 0.0
    t1 = 1.0    
    Re = 1000
    Nts = [100, 1000]
    dts = [(t1 - t0)/(Nt - 1) for Nt in Nts]
    errs = []
    for Nt, dt in zip(Nts, dts):
        # Analtical solution
        t_exact, w_exact, u_exact = nt.taylor_vortex(N, L, dt, Nt, Re, R)
        w_exact = np.absolute(w_exact)
        # Numerical solution
        (w0, W0, u0) = nt.taylor_init(N, L, R, init=False)
        (t_num, W_num, dt_max) = node.RK4_2d(nt.get_dW, W0, N, dt, Nt, Re)
        w_num = np.absolute(nf.IFFT2(np.fft.ifftshift(W_num)))
        # Error estimate
        errs.append(nt.RMS2(w_num[-1] - w_exact[-1]))
    
    slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)
    plt.figure()
    plt.loglog(dts, errs, label='m={:4.2f}'.format(3.745))
    plt.legend()
    plt.title('Convergence rate with RK4 scheme')
    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.grid()
    # Heat map
    fg1 = nt.show_heat_map(w_exact, t_exact)
    fg2 = nt.show_heat_map(w_num, t_num)

    #----- Q2(b)(iii) spatial convergence test -----#
    L = 1.5
    R = 'Auto'
    t0 = 0.0
    t1 = 0.001
    Nt = 100
    dt = (t1 - t0)/(Nt - 1)
    Re = 1000
    N_f = 256
    Ns = [512, 256, 128, 64, 32]
    for N in Ns:
        (w0, W0, u0) = nt.taylor_init(N, L, R, init=False)
        (t, W, dt_max) = node.RK4_2d(nt.get_dW, W0, N, dt, Nt, Re)
        w = nf.IFFT2(W)
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
        elif N == Ns[4]:
            dN = Ns[0]//N
            w_c4 = w
            w_f4 = w_f[:, ::dN, ::dN]
    
    # Error estimate
    errs1 = nt.RMS2(w_f1[-1] - w_c1[-1])
    errs2 = nt.RMS2(w_f2[-1] - w_c2[-1])
    errs3 = nt.RMS2(w_f3[-1] - w_c3[-1])
    errs4 = nt.RMS2(w_f4[-1] - w_c4[-1])
    errs = [errs1, errs2, errs3, errs4]

    print('error 256:{}, 128: {}, 64: {}, 32:{}'.format(errs1, errs2, errs3, errs4))
    plt.figure()
    plt.semilogy(Ns[1:], errs)
    plt.title('Sptial convergence rate')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.grid()

    #----- Q2(c) co-rotating Taylor vortex flow -----#
    def init_condition(N, n, b, L):
        c = L*n/N
        d = L*b/N
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
        c = L*n/N
        d = L*b/N
        ct1 = N//2 + b//2
        ct2 = N//2 - b//2
        vt, w_single, u = nt.taylor_vortex(n, c, dt, Nt, Re, R='Auto')
        vw = np.array([np.zeros((N, N)) for i in range(Nt)])
        vw[:, N//2, N//2] = 0
        vw[:, N//2 - n//2:N//2 + n//2, ct1 - n//2:ct1 + n//2] += w_single
        vw[:, N//2 - n//2:N//2 + n//2, ct2 - n//2:ct2 + n//2] += w_single

        return vt, vw

    L = 1.5
    R = 'Auto'
    t0 = 0.0
    t1 = 0.5
    Nt = 500
    dt = (t1 - t0)/(Nt - 1)
    Re = 100
    N = 256
    n = 128
    bs = [8, 16, 32, 64, 128]
    ln = len(bs)
    for i in range(ln):
        b = bs[i]
        t, w = co_taylor_vortex(N, n, b, L, dt, Nt, Re)
        wlim = np.max(w)
        nt.show_heat_map(w, t, vmin=-wlim, vmax=wlim)

    plt.show()