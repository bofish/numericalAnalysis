import numpy as np
import pprint

def lu_decomposition(A):
    """Performs an LU Decomposition of A (which must be square into A = LU. The function returns L and U."""
    n = len(A)

    # Create zero matrices for L and U
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity, Doolittle factorization
        L[j][j] = 1.0

        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}  
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - s1

        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / U[j][j]

    return (L, U)

def forward_substitution(L, b):
    """Performs an forward substitution of Lx = b"""
    n = len(b)

    # Create zero array for x
    x = np.array([[0.0] for i in range(n)])

    # Perform the forward substitution
    for j in range(n):
        s = np.dot(L[j][0:j],x[0:j])
        x[j] = (b[j] - s)/L[j][j]

    return x

def backward_substitution(U, b):
    """Performs an backward substitution of Ux = b"""
    n = len(b)

    # Create zero array for x
    x = np.array([[0.0] for i in range(n)])

    # Perform the backward substitution
    for j in range(n-1,-1,-1):
        s = np.dot(U[j][j:n],x[j:n])
        x[j] = (b[j] - s)/U[j][j]
    return x

def LU_solver(A, b):
    """Performs an LU Decomposition of A to solve linear system of equation Ax = b. The function returns x."""

    # Perform the LU Decomposition to obtain L, U matrix
    L, U = lu_decomposition(A)

    # Solve two subproblem, Ly = b and Ux = y
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

if __name__ == '__main__':
    A = [[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]]
    b = [[1], [2], [3], [4]]
    x = LU_solver(A, b)
    pprint.pprint(x)