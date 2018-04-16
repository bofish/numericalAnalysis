from general import *

#------ Parameter ------#
N = 10
a = -1
b = 1
x_sub = np.linspace(a,b,100)

#------Q1(a)------#

# Sampling
x_es, f_es = evenly_space_pt(a, b, N)
# print('-Evenly spaced points-')
# print('\n'.join('{0:2d} & {1:4.3f} & {2:5.4f}  \\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))

# Interpolation result
plt.figure()

f_van = polynomial_vandermode(x_es, f_es, x_sub)
plt.plot(x_sub, f_van, ':r')

lpoly = polynomial_lagrange(x_es, f_es)
plt.plot(x_sub, lpoly(x_sub), '-.g')

npoly = polynomial_newton(x_es, f_es)
plt.plot(x_sub, npoly(x_sub), '--b')

f_ori = origin_fun(x_sub, a,b)
plt.plot(x_sub, f_ori, 'k', x_es, f_es, 'or')

plt.title('Polynomial interpolation with evenly spacing')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Vandermode', 'Lagrange', 'Newton', 'Origin'])

# Error distribution
plt.figure()

poly_error = cal_error(f_ori, lpoly(x_sub))
plt.plot(x_sub, poly_error)

plt.title('Error distribution by polynomial with evenly spacing')
plt.xlabel('x')
plt.ylabel('error')

#------Q1(b)------#

# Sampling
x_cp, f_cp = chebyshev_pt(a, b, N)
# print('-Chebyshev points-')
# print('\n'.join('{0:2d} & {1:4.3f} & {2:5.4f}  \\\\'.format(j+1, x[j], f[j]) for j in range(0,N)))

# Interpolation result
plt.figure()

x_sub = np.linspace(a,b,100)

f_van = polynomial_vandermode(x_cp, f_cp, x_sub)
plt.plot(x_sub, f_van, ':r')

lpoly = polynomial_lagrange(x_cp, f_cp)
plt.plot(x_sub, lpoly(x_sub), '-.g')

npoly = polynomial_newton(x_cp, f_cp)
plt.plot(x_sub, npoly(x_sub), '--b')

f_ori = origin_fun(x_sub, a,b)
plt.plot(x_sub, f_ori, 'k', x_cp, f_cp, 'or')

plt.title('Polynomial interpolation with Chebyshev points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Vandermode', 'Lagrange', 'Newton', 'Origin'])

# Error distribution
plt.figure()
poly_error = cal_error(f_ori, lpoly(x_sub))
plt.plot(x_sub, poly_error)
plt.title('Error distribution by polynomial with Chebyshev points')
plt.xlabel('x')
plt.ylabel('error')

#------Q1(c)------#

# Interpolation result
plt.figure()

f_spl = cubic_spline(x_es, f_es, x_sub)
plt.plot(x_sub, f_spl, ':r')

lpoly = polynomial_lagrange(x_es, f_es)
plt.plot(x_sub, lpoly(x_sub), '-.g')

f_ori = origin_fun(x_sub, a,b)
plt.plot(x_sub, f_ori, 'k', x_es, f_es, 'or')

plt.title('Spline interpolation with evenly spacing')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Natrue spline', 'Lagrange polynomial', 'Origin'])

# Error distribution
plt.figure()

spl_error = cal_error(f_ori, f_spl)
plt.plot(x_sub, spl_error)

plt.title('Error distribution by spline with evenly spacing')
plt.xlabel('x')
plt.ylabel('error')

#------Q1(d)------#

# Interpolation result
plt.figure()

f_spl = cubic_spline(x_cp, f_cp, x_sub)
plt.plot(x_sub, f_spl, ':r')

lpoly = polynomial_lagrange(x_cp, f_cp)
plt.plot(x_sub, lpoly(x_sub), '-.g')

f_ori = origin_fun(x_sub, a,b)
plt.plot(x_sub, f_ori, 'k', x_cp, f_cp, 'or')

plt.title('Spline interpolation with Chebyshev points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['Natrue spline', 'Lagrange polynomial', 'Origin'])

# Error distribution
plt.figure()

spl_error = cal_error(f_ori, f_spl)
plt.plot(x_sub, spl_error)

plt.title('Error distribution by spline with Chebyshev points')
plt.xlabel('x')
plt.ylabel('error')

plt.show()