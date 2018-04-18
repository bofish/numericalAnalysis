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
    3. Output length less than len(f), sample number caused
    4. If len(df) = 100, len(f_diff)=95, means f_diff[0:94] is match to df[0:94]
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
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(M-sample_number.loc[(k,m)]+1):
        # single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(6) if p<=M-j-1]
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(sample_number.loc[(k,m)])]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    return f_diff

def backward_diff(x, f, k, m):
    '''
    Limitation:
    1. For `k`-rd derivative
    2. `h` is evenly spacing
    3. Output length less than len(f), sample number caused
    4. If len(df) = 100, len(f_diff)=95, means f_diff[5:99] is match to df[5:99]
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
    sample_number = count_sample_number(km_array, backward_diff_coeff)

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(sample_number.loc[(k,m)]-1, M):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(0, -sample_number.loc[(k,m)], -1)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    return f_diff

def central_diff(x, f, k, m):
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
        [0, 0, -1/2, 1.0e-310, 1/2, 0, 0],
        [0, 1/12, -2/3, 1.0e-310, 2/3, -1/12, 0],
        [0, 0, 1, -2, 1, 0, 0],
        [0, -1/12, 4/3, -5/2, 4/3, -1/12, 0],
        [0, -1/2, 1, 1.0e-310, -1, 1/2, 0],
        [1/8, -1, 13/8, 1.0e-310, -13/8, 1, -1/8],
        [0, 1, -4, 6, -4, 1, 0],
        [-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6],
    ])
    coeff_df = make_coeff_table_df(km_array, p_array, central_diff_coeff)
    sample_number = count_sample_number(km_array, central_diff_coeff)
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(int((sn-1)/2), int(M - (sn-1)/2)):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(int(-(sn-1)/2),int((sn+1)/2))]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    print(len(f_diff))
    return f_diff

if __name__ == '__main__':
    x = np.linspace(1,2,100)
    f = 2*np.sin(x) - np.exp(x)/4 - 1
    df = 2*np.cos(x) - np.exp(x)/4
    # df_forward = forward_diff(x, f, 1, 2)
    # print(df, df_forward)
    # df_backward = backward_diff(x, f, 1, 2)
    # print(df, df_backward)
    df_central = central_diff(x, f, 1, 4)
    print(df, df_central)
    # plt.plot(x,df, x, df_forward)
    # plt.show()
    # M = 10
    # sn = 7
    # foo = []
    # for j in np.arange((sn-1)/2, M - (sn-1)/2):
    #     # a = [x[j+p] for p in range(6) if p<=M-j-1]
    #     foo.append(j)
    # print(foo)