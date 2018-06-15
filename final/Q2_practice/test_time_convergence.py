import numerical_taylor as nt
import exact_taylor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    L = 1.5
    R = 'Auto'
    N = 64
    t0 = 0.0
    t1 = 1.0
    # Nt = 1000
    
    Re = 1000
    
    Nts = [100, 300, 700, 1000]
    dts = [(t1 - t0)/(Nt - 1) for Nt in Nts]
    errs = []
    for Nt, dt in zip(Nts, dts):
        
        
        # Analtical solution
        t_exact, w_exact, u_exact = nt.taylor_vortex(N, L, dt, Nt, Re, R)
        w_exact = np.absolute(w_exact)
        # Numerical solution
        (w0, W0, u0) = nt.taylor_init(N, L, R, init=False)
        (t_num, W_num, dt_max) = nt.RK4_2d(nt.get_dW, W0, N, dt, Nt, Re)
        # w_num = np.fft.ifft2(W_num)
        w_num = np.absolute(np.fft.ifft2(np.fft.ifftshift(W_num)))
        # w_num = np.fft.ifft2(np.fft.ifftshift(W_num))
        print(Nt, dt, dt_max)
        # Error estimate
        errs.append(nt.RMS2(w_num[-1] - w_exact[-1]))
    
    print(errs)
    slope, intercept = np.polyfit(np.log(Nts), np.log(errs), 1)
    plt.figure()
    plt.loglog(Nts, errs, label='m={:4.2f}'.format(slope))
    plt.legend()
    plt.title('Finite difference with convergence rate')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.grid()

    plt.figure()
    plt.loglog(dts, errs, label='m={:4.2f}'.format(slope))
    plt.legend()
    plt.title('Finite difference with convergence rate')
    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.grid()

    plt.figure()
    plt.plot(Nts, errs, label='m={:4.2f}'.format(slope))
    plt.legend()
    plt.title('Finite difference with convergence rate')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.grid()
    # Heat map
    fg1 = nt.show_heat_map(w_exact, t_exact)
    fg2 = nt.show_heat_map(w_num, t_num)

    plt.show()