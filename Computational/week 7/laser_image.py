# %%
import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% This is the way we'd like to write a 2D function
def gauss_2D(x, y, A, x_mean, x_varience, y_mean, y_varience):
    return ( A*exp(-1/2*(x-x_mean)**2/x_varience)
                *exp(-1/2*(y-y_mean)**2/y_varience) )

# %%
# A tip, always use a slightly different x and y size
# so you know you get the transposing of arrays correct!

data = np.load("noisy_2D_gaussian.npz")
x = data['x']
y = data['y']
x,y = np.meshgrid(x,y)  # meshgrid since image is 2D
z = data['z']
print(x.shape)
print(y.shape)
print(z.shape)

plt.contourf(x, y, z, 100)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Original 2D plot")

# %% We now have to transform our nice 2D function into one that looks like a 1D
def gauss_ravelled(t, *p):
    # *parameters here is used to forward on all position arguments to the
    # gaussian_2D, it saves rewriting the arguments out
    
    # unpack t back into x and y
    x = t[0] # is this a list?
    y = t[1]
    # Now we flatten (ravel) the model
    return gauss_2D(x, y, *p).ravel()

# Here we pack the X and Y 2D arrays into 1D ones
t = np.array([x, y])
print("Shape of t", t.shape)

# guess allows the curve_fit() to know how many parameters to expect
# since the gauss_ravelled(t,*p) has *p, which is an undefined size
guess = [1,-1/2,1,0,1]

# find the parameters using curve_fit(f(y,p), x, y_data, p0 = initial guess)
p, cov = curve_fit(gauss_ravelled, t, z.ravel(), p0 = guess)
print(p)

plt.contourf(x, y, gauss_2D(x,y,*p), 20)

# %%
# now for the star image, load the data & setup

def airy(x,y,x0,y0,I0,S,grad):
    """
    x = x-axis      [array]
    y = y-axis      [array]
    x0 = x position [float]
    y0 = y position [float]
    I0 = intensity  [float]
    S = size        [float]
    grad = gradient [float]
    """
    from scipy.special import j1
    eps = 1-20      # avoid singularities
    r = sqrt( (x-x0)**2 + (y-y0)**2 )
    r[r==0] = eps
    return I0*abs(j1(r/S)/r)**2 + x*grad
def airy_ravel(t, *p):
    """
    t = [x,y]       [3D object]
    *p = parameters [array] 
    """
    x,y = t
    return airy(x, y, *p).ravel()

# extract & show image
image = np.load("star.npz")['arr_0']
plt.imshow(image, cmap = 'gray', origin = 'lower')
print('Dimensions of image', np.shape(image))

x = np.arange(1024)     
y = x.copy()
x,y = np.meshgrid(x,y)  # 1240x1240 clear image, curve_fit airy function over this to subtract from image
t = np.array([x,y])
#t = np.vstack([X.ravel(),Y.ravel()])
print('Shape of t', t.shape)
# z = image[3]
print('Shape of z', z.shape)
guess = [512,512,1,1,0]

# %%

# find the parameters using curve_fit(f(y,p), x, y_data, p0 = initial guess)
p, cov = curve_fit(airy_ravel, t, image.ravel(), p0 = guess)
print(p)

plt.contourf(x, y, airy(x,y,*p), 20)
plt.figure()
plt.imshow(image - airy(x,y,*p), cmap = 'gray', origin = 'lower')
# %%
