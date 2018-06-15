import numerical_taylor as nt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    L = 1.5
    N = 64
    t0 = 0.0
    t1 = 1
    Nt = 500 
    dt = (t1 - t0)/(Nt - 1)
    Re = 1000
    
    # Analtical solution
    t_exact, w_exact, u_exact = nt.taylor_vortex(N, L, dt, Nt, Re)
    u_max = np.max(u_exact)
    print(u_max)
    # fg1 = nt.show_heat_map(w_exact, t_exact)
    
    plt.show()