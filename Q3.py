from numerical.calculus import *
# Parameter 
a = 6
b = 10
N = 100
x = np.linspace(a,b,N)
f = 2*np.sin(x) - np.exp(x)/4 - 1

#------Q3(a)------#

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

#------Q3(b)------#

#------Q3(c)------#

plt.show()