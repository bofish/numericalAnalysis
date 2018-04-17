import numpy as np

def forward_diff(x, f, n):
    '''
    Limitation:
    1. For `n`-rd derivative
    2. `h` is evenly spacing
    '''
    M = len(x)
    f_diff = []
    for j in range(M):
        h = x[j+1] - x[j] 
        single_pt = a[p]*f[j+p] for p in range(m)
        dnf_j = ()/h


    f_diff.append(dnf)
    return f_diff

if __name__ == '__main__':
    a = [i*2 for i in range(3)]
    a = sum(a)
    print(a)