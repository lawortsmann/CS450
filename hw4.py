# -*- coding: utf-8 -*-
# HW4.py
"""
Code for CS 450 HW 4
@version: 10.15.2015
@author: wortsma2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

# Problem 1:

def get_A(n):
    """
    Build a random matrix with eigenvalues 1 through n
    """
    B = np.random.random(size=(n,n))
    Q = np.linalg.qr(B)[0]
    D = np.diag(range(1,n+1))
    A = np.dot(np.transpose(Q),np.dot(D,Q))
    return np.matrix(A)

# Initial parameters:
n = 20
A = get_A(n)
# Q is a list of column vectors
Q = []
# Q0 is 0
Q.append(np.matrix(np.zeros((n,1))))
# Q1 is random w norm 1
q1 = np.matrix(np.random.random((n,1)))
Q.append(q1/np.linalg.norm(q1))
# b0 is 0 and a0 is 0
b = [0]
a = [0]
# store ritz values after each iteration:
ritz = []
its = []
# k = 1,2,3,...,n
for k in xrange(1,n+1):
    # A Lanczos iteration:
    uk = A.dot(Q[k])
    ak = np.dot((Q[k].getH()),uk)[0,0]
    a.append(ak)
    uk = uk - a[k]*Q[k] - b[k-1]*Q[k-1]
    bk = np.linalg.norm(uk)

    # stop if Q is spanning set
    if bk < 5e-16:
        break
    b.append(bk)
    Q.append(uk/b[k])

    # build Tk from a and b:
    Tk = np.matrix(np.diag(a[1:])+np.diag(b[1:-1],k=1)+np.diag(b[1:-1],k=-1))
    # find ritz values and add them to list
    ritz += np.linalg.eigvals(Tk).tolist()
    # also add iteration number
    its += (k*np.ones(k)).tolist()

# plot ritz values
plt.plot(ritz,its,'.')
plt.axis([0.,21.,0.,21.])
plt.show()

# Problem 2:

A = np.array([[-261., 209., -49.],[-530., 422., -98.],[-800., 631., -144.]])

def QR_i(A):
    Q,R = np.linalg.qr(A)
    return np.dot(R,Q)

def QR_algo(A,k):
    Ak = np.copy(A)
    b1 = [abs(Ak[1,0])]
    b2 = [abs(Ak[2,0])]
    b3 = [abs(Ak[2,1])]
    ks = range(k+1)
    for i in xrange(k):
        Ak = QR_i(Ak)
        b1.append(abs(Ak[1,0]))
        b2.append(abs(Ak[2,0]))
        b3.append(abs(Ak[2,1]))
    return sorted(np.diagonal(Ak),reverse=True),b1,b2,b3,ks

def shifted_QR_algo(A,k):
    Ak = np.copy(A) - np.identity(3)*4.0
    b1 = [abs(Ak[1,0])]
    b2 = [abs(Ak[2,0])]
    b3 = [abs(Ak[2,1])]
    ks = range(k+1)
    for i in xrange(k):
        Ak = QR_i(Ak)
        b1.append(abs(Ak[1,0]))
        b2.append(abs(Ak[2,0]))
        b3.append(abs(Ak[2,1]))
    return sorted(np.diagonal(Ak)+4,reverse=True),b1,b2,b3,ks

eig,b1,b2,b3,k = QR_algo(A,10)
eigshift,c1,c2,c3,k = shifted_QR_algo(A,10)
plt.semilogy(k,b1,label="A(2,1)")
plt.semilogy(k,b2,label="A(3,1)")
plt.semilogy(k,b3,label="A(3,2)")
plt.legend()
plt.show()
plt.semilogy(k,c1,label="A(2,1)")
plt.semilogy(k,c2,label="A(3,1)")
plt.semilogy(k,c3,label="A(3,2)")
plt.legend()
plt.show()
print np.linalg.eigvals(A)
print QR_algo(A,2)[0]
print shifted_QR_algo(A,2)[0]

# Problem 3:

def g1(x):
    return (x*x + 2.0)/3.0

def g2(x):
    return (3.0*x - 2.0)**0.5

def g3(x):
    return 3.0 - (2.0/x)

def g4(x):
    return (x*x - 2.0)/(2.0*x - 3.0)

def fixed_point(g,x0):
    x = x0
    xn = g(x)
    fpl = [abs(x-2.0),abs(xn-2.0)]
    while xn != x:
        x = xn
        xn = g(x)
        fpl.append(abs(xn-2.0))
    return plt.semilogy(fpl,label = g.__name__)

x = 1.555
fixed_point(g1,x)
fixed_point(g2,x)
fixed_point(g3,x)
fixed_point(g4,x)
plt.legend()
plt.show()

# Problem 4:

# Functions and their derivatives:
def fa(x):
    return x*x*x - 2.0*x - 5.0

def dfa(x):
    return 3.0*x*x - 2.0

def fb(x):
    return np.exp(-x) - x

def dfb(x):
    return -np.exp(-x) - 1

def fc(x):
    return x*np.sin(x) - 1

def dfc(x):
    return np.sin(x) + x*np.cos(x)

def fd(x):
    return x*x*x - 3.0*x*x + 3.0*x - 1.0

def dfd(x):
    return 3*x*x - 6.0*x + 3.0

def bisection(f,a,b,i=0):
    """
    Implementation of the bisection method
    """
    m = a + (b-a)/2.0
    if (b-a) < 5e-16:
        return m,i
    else:
        pa = f(a)
        pm = f(m)
        if np.sign(pa) == np.sign(pm):
            return bisection(f,m,b,i=i+1)
        else:
            return bisection(f,a,m,i=i+1)

def newton(f,df,x0, maxiterations = 100, tol = 5e-16):
    """
    Implementation of the newton method
    """
    xk = x0
    e = [xk]
    for i in xrange(maxiterations):
        dfxk = df(xk)
        if dfxk == 0.0:
            break
        d = f(xk)/dfxk
        if abs(d) < tol:
            break
        else:
            xk -= d
            e.append(xk)
    if i+1 == maxiterations:
        return
    return xk,i,abs(np.array(e)-xk)

def secant(f, x0, maxiterations = 100, tol = 5e-16):
    """
    Implementation of the secant method
    """
    xk = x0
    xkm = x0 - (1e-15)
    fxk = f(xk)
    fxkm = f(xkm)
    e = [xk]
    for i in xrange(maxiterations):
        dn = fxk - fxkm
        if dn == 0.0:
            break
        d = fxk*((xk - xkm)/dn)
        if abs(d) < tol:
            break
        xkm = xk
        xk -= d
        fxkm = fxk
        fxk = f(xk)
        e.append(xk)
    if i+1 == maxiterations:
        return
    return xk,i,abs(np.array(e)-xk)
# Bisection for each function, interval [0,3] is a good interval
print bisection(fa,0.,3.)
print bisection(fb,0.,3.)
print bisection(fc,0.,3.)
print bisection(fd,0.,3.)
# newton for each function, x0 = 1.1 is a good starting value
print newton(fa,dfa,1.1)[:2]
print newton(fb,dfb,1.1)[:2]
print newton(fc,dfc,1.1)[:2]
print newton(fd,dfd,1.1)[:2]
# secant for each function, x0 = 1.1 is a good starting value
print secant(fa,1.1)[:2]
print secant(fb,1.1)[:2]
print secant(fc,1.1)[:2]
print secant(fd,1.1)[:2]
# Using library functions for fa:
print so.bisect(fa,0.,3.)
print so.newton(fa,1.1,fprime=dfa)
print so.newton(fa,1.1)
# Using library functions for fb:
print so.bisect(fb,0.,3.)
print so.newton(fb,1.1,fprime=dfb)
print so.newton(fb,1.1)
# Using library functions for fc:
print so.bisect(fc,0.,2.)
print so.newton(fc,1.1,fprime=dfc)
print so.newton(fc,1.1)
# Using library functions for fd:
print so.bisect(fd,0.,3.)
print so.newton(fd,1.1,fprime=dfd)
print so.newton(fd,1.1)
# Plot convergence rate for newton method:
plt.semilogy(newton(fa,dfa,1.1)[2], label = 'fa')
plt.semilogy(newton(fb,dfb,1.1)[2], label = 'fb')
plt.semilogy(newton(fc,dfc,1.1)[2], label = 'fc')
plt.semilogy(newton(fd,dfd,1.1)[2], label = 'fd')
plt.legend()
plt.show()
# Plot convergence rate for secant method:
plt.semilogy(secant(fa,1.1)[2], label = 'fa')
plt.semilogy(secant(fb,1.1)[2], label = 'fb')
plt.semilogy(secant(fc,1.1)[2], label = 'fc')
plt.semilogy(secant(fd,1.1)[2], label = 'fd')
plt.legend()
plt.show()

def convergence(e):
    """
    Estimate the rate of convergence of a root finding method
    """
    t = []
    for ri in xrange(300):
        r = 1.0 + (ri/100.0)
        c = [e[i]/(e[i-1]**r) for i in xrange(1,len(e)-1)]
        t.append((r,np.var(c),np.mean(c)))
    minr = min(t,key=lambda x: x[1])
    return minr[0],minr[2]
# Print the convergence of newton:
print convergence(newton(fa,dfa,1.1)[2])
print convergence(newton(fb,dfb,1.1)[2])
print convergence(newton(fc,dfc,1.1)[2])
print convergence(newton(fd,dfd,1.1)[2])
# Print the convergence of secant:
print convergence(secant(fa,1.1)[2])
print convergence(secant(fb,1.1)[2])
print convergence(secant(fc,1.1)[2])
print convergence(secant(fd,1.1)[2])
