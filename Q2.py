import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
# Parameters
a = 1
N = 1024
x = np.linspace(-pi, pi, N)
g = np.exp(-a*x**2)
# G = sqrt(pi/a)*np.exp(-pi**2*X**2/a)
f_m = []
f = []
x_f = []
for x_m in x:
    for m in range(-10,10):
        f_m.append(np.exp(-a*(x_m-2*pi*m)**2))
        print(len(f_m))
    f.append(np.sum(f_m))
print(len(f))
plt.plot(x, f)
plt.plot(x, g)
plt.show()