from numerical_taylor import taylor_vortex
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    L = 1.5
    N = 64
    t0 = 0.0
    t1 = 3
    Nt = 700
    dt = (t1 - t0)/(Nt - 1)
    Re = 1
    
    # Analtical solution
    t_exact, w_exact = taylor_vortex(N, L, dt, Nt, Re)
    print(t_exact)
    print(w_exact)
    plt.figure(1)
    p = int(321)
    for i in np.linspace(1, Nt, 6):
        i = int(i)-1
        print(i)
        plt.subplot(p)
        ax = sns.heatmap(w_exact[i], cmap="YlGnBu") 
        plt.title(i) 
        p += 1

    plt.show()