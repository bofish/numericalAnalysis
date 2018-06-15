import numerical_taylor as nt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    L = 1.5
    N = 16
    t0 = 0.0
    t1 = 1.0
    Nt = 150
    dt = (t1 - t0)/(Nt - 1)
    
    W0_rand = nt.init_process(np.random.randn(N, N) + 1j*np.random.randn(N, N))
    
    vdt_max = []
    vRe = range(10,500,50)
    for Re in vRe:
        print(Re)
        (vt, vW, dt_max) = nt.RK4_2d(nt.get_dW, W0_rand, N, dt, Nt, Re)
        vdt_max.append(dt_max)
    plt.plot(vRe, np.real(vdt_max))
    plt.show()

