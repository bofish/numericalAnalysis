import numpy as np

# Step 1: Discretize x
n = 11
x = np.linspace(0.0, 1.0, n)
dx = x[1] - x[0]

# Step 2: Discretize ODE by 2nd order central difference scheme
#         to form ODE into $a_iy_{i-1} + b_iy_i + c_iy_{i+1} = d_i$
a_i = 1/dx**2 - 2/dx
b_i = -2/dx**2 + 4
c_i = 1/dx**2 + 2/dx
d_i = x_i

# Step 6: Set up the linear system, return a banded structure matrix
def make_band_mat():
    y = [[0.0] for i in range(n)]
    # Create a zero matrix with n*n dimension
    A = [[0.0] * n for i in range(n)]
    for rpw in range(n):
        A[row][i:i+2] = 1
    
    for j in range(1,M-1):
        row = j-1
        coeff_mat[row, j-1] = delta[j-1]/6
        coeff_mat[row, j] = (delta[j-1]+delta[j])/3
        coeff_mat[row, j+1] = delta[j]/6
        f_mat[row, 0] = delta_f[j] - delta_f[j-1]
    # For the nature spline, we given the conditin which is f1" = fn" = 0
    coeff_mat = np.delete(coeff_mat, M-1, 1)
    coeff_mat = np.delete(coeff_mat, 0, 1)