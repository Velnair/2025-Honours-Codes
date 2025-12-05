# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% Define a simple cubic model
def quad_model(x, a, b, c):
    return x**2 * a + x * b + c

# %%
# First we need to set up our problem and make some fake noisy data.
# This is useful as we know the true values we put in and later can see
# if we get close the right answer
true_p = [2, 0.3, 4] # true answer @ x = 1.1

D_x = np.linspace(0, 5, 100)
true_y = quad_model(D_x, *true_p)
# Let's add some random normally distributed noise
# First we set the random seed to the same value so we get the same data each
# time, you can not do this and get different samples each run if you wish
np.random.seed(0) 
# add zero mean noise with a std deviation of 50
std = 10
D_y = true_y + np.random.normal(0, std, len(D_x))

plt.scatter(D_x, D_y, label='Noisy data values', c='g')
plt.scatter(D_x, true_y, 50, label='True values', marker='+')
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("y")

# %%
x = D_x.copy()
# By setting cov=True it will return the fitted parameters as well as the
# covariance matrix for us to print later

fit_p_poly, fit_cov_poly = np.polyfit(D_x, D_y, len(true_p)-1, cov=True)
fit_p_curv, fit_cov_curv = curve_fit(quad_model, D_x, D_y, p0 = None) # p0 is dependent on the data provided 

print(f'polyfit parameters: {fit_p_poly}')
print(f'curvefit parameters: {fit_p_curv}')
print(f'true parameters: {true_p}')

fig, axs = plt.subplots(2, 1, sharex=True)
plt.sca(axs[0])
plt.scatter(D_x, D_y, label='Noisy data values', c='g')
plt.plot(x, np.polyval(true_p, x), label='True')
plt.plot(x, np.polyval(fit_p_poly, x), label='numpy.polyfit', ls='--', lw=3)
plt.plot(x, quad_model(x,*fit_p_curv), label='scipy.curve_fit',color = 'y', ls=':', lw=3)
plt.legend()
plt.grid()
plt.ylabel("y")

plt.sca(axs[1])
plt.scatter(D_x, D_y - np.polyval(fit_p_poly, D_x), label = 'polyfit')
plt.scatter(D_x, D_y - np.polyval(fit_p_curv, D_x), label = 'curvefit')
plt.legend()
plt.xlabel("x")
plt.ylabel("Residual")
plt.grid()

# %% Make a correllation matrix
# convert covariance matrix to correlation matrix
covariance = np.array([[1.0,  1.0,  8.1],
                       [1.0, 16.0, 18.0],
                       [8.1, 18.0, 81.0]])

diag = np.sqrt(np.diag(np.diag(covariance)))
gaid = np.linalg.inv(diag)
correllation = gaid @ covariance @ gaid

# %% Print our fit data out
print("Fitted polynomial coefficients")

for i, (p, var) in enumerate(zip(fit_p_poly, np.diag(fit_cov_poly))):
    print(f"p[{i}] = {p:.2g} ± {np.sqrt(var):.2g} (True = {true_p[i]})")
    
print()
print("Covariance matrix")
for i in range(fit_cov_poly.shape[0]):
    for j in range(fit_cov_poly.shape[1]):
        print(f"{fit_cov_poly[i,j]:+.2f}", end=" ")
    print()

print()
print("Curvefit coefficients")

for i, (p, var) in enumerate(zip(fit_p_curv, np.diag(fit_cov_curv))):
    print(f"p[{i}] = {p:.2g} ± {np.sqrt(var):.2g} (True = {true_p[i]})")
    
print()

print("Covariance matrix")
for i in range(fit_cov_curv.shape[0]):
    for j in range(fit_cov_curv.shape[1]):
        print(f"{fit_cov_curv[i,j]:+.2f}", end=" ")
    print()
# %%
