import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from scipy.special import factorial
from scipy.integrate import tplquad
from scipy.integrate import nquad

def legendre(m,l,x):
#Compute the assosciated legendre polynomial of degree m and l.
# legendre(m,l,x)  returns the laguerre polynomial
# of degree m, l where m and l are positive integers and x is a 1d numpy array.
    
    
# Check whether l is a non-negative integer
    if np.logical_or(round(l) - l!= 0, l < 0):
        #return print('l must be a positive integer')
        raise ValueError('l must be a positive integer') 
    
# Check whether m is an integer and abs(m)<=l
   
    if np.logical_or(round(m)- m!= 0, abs(m) > l):
        #return print('m must be a integer with abs(m)<= l')
        
        raise ValueError('m must be a positive integer')   
        
        
# Calculate the leg(m=0,l=0),leg(m=0,l=1) and leg(m=0,l=2) as p1,p2 and p3 
    p1 = np.array(1)
    p2 = np.array(x)
    L = 2
    M = 0
    p3 = ((2*L-1)*x*p2-(L+M-1)*p1)/(L-M)
    
    breakpoint()
    
# Calculate the values of the polymomials  from leg(m=0,l=0),leg(m=0,l=1) and leg(m=0,l=2)to
# iterate to get leg(m=m,l=m), leg(m=m,l=m+1) and leg(m=m,l=m+2)
    
    if abs(m)>0:
        for ii in range(1, abs(m)+1):
            M = ii
            L = ii
            p1 = ((L-M+1)*x*p2-(L+M-1)*p1)/np.sqrt(1-x**2)
            L = L + 1
            p2 = ((L-M+1)*x*p3-(L+M-1)*p2)/np.sqrt(1-x**2)
            L = L + 1
            p3 =((2*L-1)*x*p2 - (L+M-1)*p1)/(L-M)
            
#If leg(m=m,l=m)or leg(m=m,l=m+1) or leg(m=m,l=m+2)are needed just use the values calculate
# previously
# If l> m+2 then simply interate using leg(m=m,l=m+1) and leg(m=m,l=m+2) as starting
# points to get leg(m=m,l=l)  
   
    if abs(m)==l:
        p = p1.copy()
    elif abs(m) == l-1:
        p = p2.copy()
    elif abs(m) == l-2:
        p = p3.copy()
 
    else:
        pa1 = p2.copy()
        pa2 = p3.copy()
    
        for ii in range(abs(m)+3, l+1):
            L = ii
            M = m
            patemp =((2*L-1)*x*pa2 - (L+M-1)*pa1)/(L-M);
            pa1 = pa2.copy()
            pa2 = patemp.copy()

    
        p = patemp.copy()
        
# If abs(m) < 0 simply use leg(m=abs(m),l=l) to calcuatle leg(m = -abs(m),l=l)
    
    if m < 0:
        n = abs(m);
        p = ((-1)**n)*(factorial(l-n)/factorial(l+n))*p;


    
    return p
    
def spherical_harmonic(theta,phi,l,m):
# function y = spherical_harmonic(theta,phi,m,l)
# A function to calculate the spherical harmonic for angles theta and phi
# and order m and l

    const = np.sqrt(((2*l+1)*factorial(l-m))/(4*np.pi*factorial(l+m)))

    y = const*legendre(m,l,np.cos(theta))*np.exp(1j*m*phi)
  
    return y

def laguerre(n,alpha,x):
#LAGUERRE Compute the LAGUERRE polynomial of degree n.
# h = laguerre(n, alpha, x) returns the laguerre polynomial
# of degree n, alpha in x.


#Check whether n is a non-negative integer
    if np.logical_or(round(n)- n!= 0, n < 0):
        #return print('n must be a positive integer')
        raise ValueError('n must be a positive integer') 
    
# Set up a vector of the same size as final vecto
    lp = np.zeros(n+1)
    
# if either of the first 2 laguerre polynomials just state them
    if n==0:
        lp = np.ones(1)
    elif n== 1:
        lp = [alpha+1.0,-1.0]
    elif n>1:
        lp_l = np.zeros(n+1)
        lp_u = np.zeros(n+1)
        lp_l[0] = 1.0
        lp_u[0] = alpha + 1
        lp_u[1] = -1
        
        for ii in range(2, n+1): 
            lp = (2+(alpha-1)/ii)*lp_u-1/ii*np.concatenate(([0],lp_u[0:n])) -(1+(alpha-1)/ii)*lp_l;
            # Remember with indice addressing this mean from 0 to less than n
            # Now reset the polynomials
            lp_l = lp_u.copy()
            lp_u = lp.copy()
    lp = np.flipud(lp)
 # Now evaluate full vector h(n,x)
    res = np.polyval(lp,x)
    return res

def hydrogenwf(r,theta,phi,n,l,m):
# hydrogenwf(r,theta,phi,n,l,m)
# calculates the wave function of the hydrogen atom for spherical
# coordinates r,theta,phi and order n,l,m

    a0 = 5.29e-11  # This is the Bohr radius in m
    
    p = 2*r/(n*a0)

    const = np.sqrt(((2/(n*a0))**3)*factorial(n-l-1)/(2*n*factorial(n+l)))


    psi = const*np.exp(-p/2)*(p**l)*laguerre(n-l-1,2*l+1,p)*spherical_harmonic(theta,phi,l,m)
    
    return psi

