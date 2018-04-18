import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_coeff_table_df(km_array, p_array, diff_coeff):
    tuples = list(zip(*km_array))
    index = pd.MultiIndex.from_tuples(tuples, names=['k', 'm'])
    coeff_df = pd.DataFrame(diff_coeff, columns=p_array, index=index)
    coeff_df.columns.name = 'p'
    return coeff_df
def count_sample_number(km_array, diff_coeff):
    sample_number = np.count_nonzero(diff_coeff, axis=1)
    tuples = list(zip(*km_array))
    index = pd.MultiIndex.from_tuples(tuples, names=['k', 'm'])
    sample_number_s = pd.Series(sample_number, index=index)

    return sample_number_s

def forward_diff(x, f, k, m):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
               [1, 2, 1, 2, 1, 2, 1, 2]]
    p_array = [0, 1, 2, 3, 4, 5]
    forward_diff_coeff = np.array([
        [-1, 1, 0, 0, 0, 0],
        [-3/2, 2, -1/2, 0, 0, 0],
        [1, -2, 1, 0, 0, 0],
        [2, -5, 4, -1, 0, 0],
        [-1, 3, -3, 1, 0, 0],
        [-5/2, 9, -12, 7, -3/2, 0],
        [1, -4, 6, -4, 1, 0],
        [3, -14, 26, -24, 11, -2],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, forward_diff_coeff)
    sample_number = count_sample_number(km_array, forward_diff_coeff)
    
    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[1])**2/M

    for j in range(M-sample_number.loc[(k,m)]+1):
        # single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(6) if p<=M-j-1]
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(sample_number.loc[(k,m)])]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    print(len(f_diff))
    return f_diff

def backward_diff(x, f, k):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
                [1, 2, 1, 2, 1, 2, 1, 2]]
    p_array = [-5, -4, -3, -2, -1, 0]
    backward_diff_coeff = np.array([
        [0, 0, 0, 0, -1, 1],
        [0, 0, 0, 1/2, -2, 3/2],
        [0, 0, 0, 1, -2, 1],
        [0, 0, -1, 4, -5, 2],
        [0, 0, -1, 3, -3, 1],
        [0, 3/2, -7, 12, -9, 5/2],
        [0, 1, -4, 6, -4, 1,],
        [-2, 11, -24, 26, -14, 3],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, backward_diff_coeff)


    return coeff_df

def central_diff(x, f, k):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    '''
    # Table of Coefficients 
    km_array = [[1, 1, 2, 2, 3, 3, 4, 4],
               [2, 4, 2, 4, 2, 4, 2, 4]]
    p_array = [-3, -2, -1, 0, 1, 2, 3]
    central_diff_coeff = np.array([
        [0, 0, -1/2, 0, 1/2, 0, 0],
        [0, 1/12, -2/3, 0, 2/3, -1/12, 0],
        [0, 0, 1, -2, 1, 0, 0],
        [0, -1/12, 4/3, -5/2, 4/3, -1/12, 0],
        [0, -1/2, 1, 0, -1, 1/2, 0],
        [1/8, -1, 13/8, 0, -13/8, 1, -1/8],
        [0, 1, -4, 6, -4, 1, 0],
        [-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, central_diff_coeff)

    return coeff_df

if __name__ == '__main__':
    x = np.linspace(1,2,100)
    f = 2*np.sin(x) - np.exp(x)/4 - 1
    df = 2*np.cos(x) - np.exp(x)/4
    df_forward = forward_diff(x, f, 1, 2)
    print(df, df_forward)
    # plt.plot(x,df, x, df_forward)
    # plt.show()
    # M = len(x)
    # for j in range(M):
    #     a = [x[j+p] for p in range(6) if p<=M-j-1]
    #     print(a)