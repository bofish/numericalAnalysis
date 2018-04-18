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

def cal_error(f, f_hat):
    return np.sqrt((f-f_hat)**2)

def forward_diff(x, f, k, m, diff_origin=None):
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
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(M-sn+1):
        # single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(6) if p<=M-j-1]
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(sn)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[0:M-sn+1])
    error = cal_error(diff_origin[0:M-sn+1], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def backward_diff(x, f, k, m, diff_origin=None):
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
    sn = sample_number.loc[(k,m)]

    M = len(x)
    f_diff = []
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    for j in range(sn-1, M):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(0, -sn, -1)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[sn-1:M])
    error = cal_error(diff_origin[sn-1:M], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def central_diff(x, f, k, m, diff_origin=None):
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
    side_len = int((sn-1)/2)
    for j in range(side_len, M - side_len):
        single_pt = [coeff_df.loc[(k,m),p]*f[j+p] for p in range(-side_len, side_len+1)]
        dnf_j = sum(single_pt)/h**k
        f_diff.append(dnf_j)
    f_diff = np.array(f_diff)
    x_paired = np.array(x[side_len: M - side_len])
    error = cal_error(diff_origin[side_len: M - side_len], f_diff) if diff_origin is not None else ['Without derivative of f(x) inputed']
    return x_paired, f_diff, error

def trapezoidal_integ(x, f):
    M = len(x)
    h = np.sqrt(x[-1] - x[0])**2/(M-1)
    f_integ = ((f[0] + f[-1])/2 + sum(x[1:-2]))*h
    return f_integ
    
if __name__ == '__main__':
    # Parameter 
    a = 6
    b = 10
    N = 100
    x = np.linspace(a,b,N)
    f = 2*np.sin(x) - np.exp(x)/4 - 1

    # Analytical solution
    df_origin = 2*np.cos(x) - np.exp(x)/4

    # Numerical solution
    k = 1
    m = 2
    x_forward, df_forward, error_forward = forward_diff(x, f, k, m, df_origin)
    x_backward, df_backward, error_backward = backward_diff(x, f, k, m, df_origin)
    x_central, df_central, error_central = central_diff(x, f, k, m, df_origin)

    # Differentiation Result
    plt.figure()
    plt.plot(x, df_origin, '-k')
    plt.plot(x_forward, df_forward, ':r')
    plt.plot(x_backward, df_backward, '-.g')
    plt.plot(x_central, df_central, '--b')
    # plt.plot(x[[0,-1]], df_origin[[0,-1]], 'ok')
    # plt.plot(x_forward[[0,-1]], df_forward[[0,-1]], 'or')
    # plt.plot(x_backward[[0,-1]], df_backward[[0,-1]], 'og')
    # plt.plot(x_central[[0,-1]], df_central[[0,-1]], 'ob')
    plt.legend(['Origin', 'Forward', 'Backward', 'Central'])
    plt.title('Differentiation Result')
    plt.xlabel('x')
    plt.ylabel('Frist derivative of f(x)')

    # Error Result
    plt.figure()
    plt.plot(x_forward, error_forward, ':r')
    plt.plot(x_backward, error_backward, '-.g')
    plt.plot(x_central, error_central, '--b')
    plt.legend(['Forward', 'Backward', 'Central'])
    plt.title('Error Result')
    plt.xlabel('x')
    plt.ylabel('Error')

    plt.show()
    