import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy

def cn(n):
   c = y*np.exp(-1j*2*n*np.pi*time/period)
   return c.sum()/c.size

def f(x, Nh):
   f = np.array([cn(i)*np.exp(1j*2*i*np.pi*x/period)/(2*np.pi) for i in range(1,Nh+1)])
   return f.sum()

time = np.linspace(-1*np.pi, 1*np.pi, 300)
y = np.exp(-1*time**2)
period = 2*np.pi
y2 = np.array([f(t,1000).real for t in time])

plt.plot(time, y, label='y')
plt.plot(time, y2, label='y2')
plt.legend()
plt.show()