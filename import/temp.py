import numpy
import matplotlib.pyplot as plt
import scipy

def f(x, dw, st, dt):
   return numpy.sinc(dw*(dt-x)/2)**2 * numpy.exp(-x**2/4/st**2)/st

t=numpy.arange(100)*0.5e-13
plt.plot(t,f(t,1e11,1e-12,0),label='1e11,0',color='red')
plt.plot(t,f(t,1e12,1e-12,0),label='1e12,0',color='blue')
plt.plot(t,f(t,1e13,1e-12,0),label='1e13,0',color='green')
plt.plot(t,f(t,1e11,1e-12,1e-12),ls='-.',label='1e11,x',color='red')
plt.plot(t,f(t,1e12,1e-12,1e-12),ls='-.',label='1e12,x',color='blue')
plt.plot(t,f(t,1e13,1e-12,1e-12),ls='-.',label='1e13,x',color='green')

plt.legend()
plt.show()
