# %%
import numpy as np
import matplotlib.pyplot as plt

# %% This is the way we'd like to write a 2D function
def polynomial_2D(x, y, ax, bx, cx, ay, by, cy):
    return (
        (ax*x**3 + bx*x**2 + cx*x) + (ay*y**3 + by*y**2 + cy*y)
    )

# %%
# A tip, always use a slightly different x and y size
# so you know you get the transposing of arrays correct!
Nx = 100
Ny = 101

x = np.linspace(-10, 10, Nx)
y = np.linspace(-10, 10, Ny) 

p = [0, 1, 0, 0.1, 1, 0.2] # Some randomly chosen parameters
X, Y = np.meshgrid(x, y) # Generate the 2D array of x and y coordinates
Z = polynomial_2D(X, Y, *p)
plt.contourf(x, y, Z, 20)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Original 2D plot")

# %% We now have to transform our nice 2D function into one that looks like a 1D
def polynomial_ravelled(t, *parameters):
    # *parameters here is used to forward on all position arguments to the
    # gaussian_2D, it saves rewriting the arguments out
    
    # unpack t back into x and y
    x = t[0] 
    y = t[1]
    # Now we flatten (ravel) the model
    return polynomial_2D(x, y, *parameters).ravel()

# Here we pack the X and Y 2D arrays into 1D ones
t = np.array((X, Y))
print("Shape of t", t.shape)
Z_ravelled = polynomial_ravelled(t, *p)
print("Shape of Z", Z.shape)
print("Shape of Z_ravelled", Z_ravelled.shape)
# Note that Z_ravelled is a 1D array with the same total number of elements
# as the original Z

# To make a 2D plot we need to reshape it
# This should look exactly like out initial 2D plot above
# Note! That with reshaping the size of Y goes first!
plt.contourf(x, y, Z_ravelled.reshape(Ny, Nx), 20)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Ravelled and unravelled");

# The plots and arrays should be exactly the same
print("Z == Z_ravelled: ", np.all(Z_ravelled.reshape(Ny, Nx) == Z))

