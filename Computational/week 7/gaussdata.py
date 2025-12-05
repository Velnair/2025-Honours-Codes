#%%
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

D_x, D_y = np.loadtxt("measured_data.dat")
Gauss = lambda x, a, b, c, d: a*exp(-(x-b)**2/(2*c**2)) + d
two_Gauss = lambda x,a,b,c,d,e,f,g: a*exp(-(x-b)**2/(2*c**2)) + d*exp(-(x-e)**2/(2*f**2)) + g
#%%
# Assume data consists of 1 Gaussian

p1, cov = curve_fit(Gauss, D_x, D_y, p0 = [1.5,530,1,1.5])
print(p1)

# Assume data consists of 2 Gaussians

p2, cov = curve_fit(two_Gauss, D_x, D_y, p0 = [1,530,1,1,525,3,1.5])
print(p2)

plt.plot(D_x,D_y, label = 'Data')
plt.plot(D_x,two_Gauss(D_x,*p2), color = 'red', label = '2 Gaussian')
plt.plot(D_x,Gauss(D_x,*p1), color = 'green', label = '1 Gaussians')
plt.xlabel('D_x')
plt.ylabel('D_y')
plt.legend()
# %%
