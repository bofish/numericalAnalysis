from numerical.calculus import *
#------Q3(a)------#
# Parameter 
a = 6
b = 10
N = 50
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
plt.title('Differentiation Result (N={})'.format(N))
plt.xlabel('x')
plt.ylabel('Frist derivative of f(x)')

# Error Result
plt.figure()
plt.plot(x_forward, error_forward, ':r')
plt.plot(x_backward, error_backward, '-.g')
plt.plot(x_central, error_central, '--b')
plt.legend(['Forward', 'Backward', 'Central'])
plt.title('Differentiation Error Result (N={})'.format(N))
plt.xlabel('x')
plt.ylabel('Error')

#------Q3(b)------#
# Parameter 
a = -1
b = 1
N_array = np.arange(51, 101, 2)
# N_array = [51, 101, 201, 301, 401, 501]
val_analytical = []
val_trapezoidal = []
val_simpson = []
val_gauss = []
err_trapezoidal = []
err_simpson = []
err_gauss = []

for N in N_array:
    x = np.linspace(a,b,N)
    f = 2*np.sin(x) - np.exp(x)/4 - 1
    
    # Analytical solution
    sf_origin = -2*np.cos(b) - np.exp(b)/4 - b - (-2*np.cos(a) - np.exp(a)/4 - a)

    # Numerical solution
    sf_trapezoidal, error_trapezoidal = trapezoidal_integ(x, f, sf_origin)
    sf_simpson, error_simpson = simpson13_integ(x, f, sf_origin)
    sf_gauss, error_gauss = gauss_integ(x, f, sf_origin)
    
    err_trapezoidal.append(error_trapezoidal)
    err_simpson.append(error_simpson)
    err_gauss.append(error_gauss)
    val_analytical.append(sf_origin)
    val_trapezoidal.append(sf_trapezoidal)
    val_simpson.append(sf_simpson)
    val_gauss.append(sf_gauss)

    if N % 100 == 1:
        print(N)

for i in range(len(N_array)):
    print('{0:3d} & {1:7.6f} & {2:7.6f} & {3:3.2e} & {4:7.6f} & {5:3.2e} & {6:7.6f} & {7:3.2e}'.format(N_array[i], val_analytical[i], val_trapezoidal[i], err_trapezoidal[i], val_simpson[i], err_simpson[i], val_gauss[i], err_gauss[i]))
plt.figure()
plt.plot(N_array, err_simpson, ':r')
plt.plot(N_array, err_trapezoidal, '-.g')
plt.plot(N_array, err_gauss, '--b')
plt.legend(['Simson 1/3', 'Trapezoidal', 'Gauss'])
plt.title('Compare Integration Error of Simpson 1/3, Trapezoidal and Gauss')
plt.xlabel('N (odd number)')
plt.ylabel('Error')


#------Q3(c)------#
# Parameter
a = -6.5
b = -3
N = 500
x = np.linspace(a,b,N)
f = 2*np.sin(x) - np.exp(x)/4 - 1
x_0 = -5
x_1 = -6

# root-searching
# Numerical solution
root_bi = bisection_root(x_0, x_1, fsFun)
root_secant = secant_root(x_0, x_1, fsFun)
root_newton = newton_root(x_0, fsFun, dfsFun)
print(root_bi)
print(root_secant)
print(root_newton)

# Optimization solution
x_Res = fsolve(fsFun, -3.5)[0]
print('{0:4.15f}'.format(x_Res))

# Graphic solution
root_graphic = [-3.6689, -5.7591]
plt.figure()
plt.plot(x, f, x, np.zeros_like(x), '--k')
plt.plot(root_graphic,fsFun(root_graphic), 'ok')
plt.title('Graphic solution')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()