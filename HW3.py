# -*- coding: utf-8 -*-
# HW3.py
"""
Code for CS 450 HW 3
@version: 09.28.2015
@author: wortsma2
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem 1

def f1(x):
    return 1.0/(1.0 + 25.0*(x**2.0))

def Uniform(N, m = 200):
    """
    Uniform sampling, solves exact system of degree N.
    Implements np.linalg.solve, Returns max error
    """
    pN = np.empty(N+1)
    A = np.empty((N+1,N+1))
    for j in xrange(N+1):
        xi = (j / (0.5*N)) - 1.0
        pN[j] = f1(xi)
        for i in xrange(N+1):
            A[j,i] = xi**float(i)
    c = np.linalg.solve(A,pN)
    # find max residual for 200 pts
    deltaX = 2.0/(m-1.0)
    Ares = np.empty((m,N+1))
    pNres = np.empty(m)
    for i in xrange(m):
        xi = -1.0 + i*deltaX
        pNres[i] = f1(xi)
        for p in xrange(N+1):
            Ares[i,p] = xi**float(p)

    eN = max(abs((np.dot(Ares, c) - pNres))/abs(pNres))

    return eN

def Chebyshev(N, m = 200):
    """
    Chebyshev sampling, solves exact system of degree N.
    Implements np.linalg.solve, Returns max error
    """
    pN = np.empty(N+1)
    A = np.empty((N+1,N+1))
    for j in xrange(N+1):
        xi = np.cos((np.pi*j)/float(N))
        pN[j] = f1(xi)
        for i in xrange(N+1):
            A[j,i] = xi**float(i)
    c = np.linalg.solve(A,pN)
    # find max residual for 200 pts
    deltaX = 2.0/(m-1.0)
    Ares = np.empty((m,N+1))
    pNres = np.empty(m)
    for i in xrange(m):
        xi = -1.0 + i*deltaX
        pNres[i] = f1(xi)
        for p in xrange(N+1):
            Ares[i,p] = xi**float(p)

    eN = max(abs((np.dot(Ares, c) - pNres))/abs(pNres))

    return eN

def LeastSquares(N, returnC = False, m = 200):
    """
    Uniform sampling, minimizes least squares for m sample points,
    degree N. Implements np.linalg.lstsq, Returns max error.
    Returns coefficients of fit if returnC = True
    """
    deltaX = 2.0/(m-1.0)
    A = np.empty((m,N+1))
    pN = np.empty(m)
    for i in xrange(m):
        xi = -1.0 + i*deltaX
        fi = f1(xi)
        for p in xrange(N+1):
                A[i,p] = xi**float(p)
        pN[i] = fi
    c = np.linalg.lstsq(A,pN)[0]
    eN = max(abs((np.dot(A, c) - pN))/abs(pN))
    if returnC:
        return c
    else:
        return eN

# Get coefficients of fit for N = 10, m = 200
print LeastSquares(10, returnC = True)

eN_1 = []
eN_2 = []
eN_3 = []
# Error for N = 2,3,...,40
for N in xrange(2,41):
    eN_1.append(Uniform(N))
    eN_2.append(Chebyshev(N))
    eN_3.append(LeastSquares(N))
# Plot on semilogy
plt.semilogy(range(2,41),eN_1)
plt.semilogy(range(2,41),eN_2)
plt.semilogy(range(2,41),eN_3)
plt.show()

# Problem 2:

A2a = [[1],[2]]
b2a = [[1],[1]]
ca = np.linalg.lstsq(A2a,b2a)[0]
print '-------------------'
print 'projection:'
print np.dot(A2a,ca)
print 'orthogonal:'
print b2a - np.dot(A2a,ca)
print '-------------------'
A2b = [[1],[2],[3]]
b2b = [[1],[1],[1]]
cb = np.linalg.lstsq(A2b,b2b)[0]
print 'projection:'
print np.dot(A2b,cb)
print 'orthogonal:'
print b2b - np.dot(A2b,cb)
print '-------------------'
A2c = [[1,4],[2,5],[3,6]]
b2c = [[1],[1],[1]]
cc = np.linalg.lstsq(A2c,b2c)[0]
print 'projection:'
print np.dot(A2c,cc)
# Note: orthogonal is 0
print 'orthogonal:'
print b2c - np.dot(A2c,cc)
print '-------------------'
A2d = [[1,5,-4],[2,4,-2],[3,3,0],[4,2,2]]
b2d = [[1],[1],[1],[1]]
cd = np.linalg.lstsq(A2d,b2d)[0]
print 'projection:'
print np.dot(A2d,cd)
# Note: orthogonal is 0
print 'orthogonal:'
print b2d - np.dot(A2d,cd)
print '-------------------'

# Problem 4:

def LU(A):
    """
    Solution to problem 4 a
    Returns LU w/o pivoting
    Does not modify A, will return 'Singular' if pivioting is needed
    """
    L = np.identity(len(A))
    U = 1.0*np.copy(A)
    m = len(A)
    n = len(A[0])
    for k in xrange(m - 1):
        # Stop if pivoting is needed or singular:
        if U[k][k] == 0:
            return 'Singular'
        for i in xrange(k+1,m):
            factor = (U[i][k]/U[k][k])
            L[i][k] += factor
            for j in xrange(k, n):
                # Ensures lower part is 0:
                if j == k:
                    U[i][j] -= U[i][j]
                else:
                    U[i][j] -= U[k][j]*factor
    return L,U

def front_Solve(A,b):
    """
    Solution to 4 b
    Returns x where Ax = b, A is lower
    Does not modify A or b
    """
    m = len(A)
    n = len(b)
    x = 1.0*np.copy(b)
    for i in xrange(m):
        # Subtract known x's:
        for j in xrange(i):
            x[i] -= A[i][j]*x[j]
        # Divide and store:
        x[i] = x[i]/A[i][i]
    return x

def back_Solve(A,b):
    """
    Solution to 4 b
    Returns x where Ax = b, A is upper
    Does not modify A or b
    """
    m = len(A)
    n = len(b)
    x = 1.0*np.copy(b)
    for i in xrange(m):
        # Iterating from bottom row to top row:
        k = (m - 1) - i
        # Subtract known x's:
        for j in xrange(k+1,n):
            x[k] -= A[k][j]*x[j]
        # Divide and store:
        x[k] = x[k]/A[k][k]
    return x

def LU_solve(lu, b):
    """
    Solution to 4 c
    Returns x where Ax = b, given the L/U of A
    implements front_Solve and back_Solve
    """
    L,U = lu
    y = front_Solve(L,b)
    x = back_Solve(U,y)
    return x

def Sherman_Morrison(u,v,b,lu):
    """
    For problems 4 e f g h
    Solve:
    (A + u v^T) x = b
    and return x given the L/U for A
    """
    # Calculate needed inverse products using LU:
    xu = LU_solve(lu,u)
    xb = LU_solve(lu,b)
    # Calculate other stuff:
    c1 = np.dot(xu,v)
    c2 = np.dot([[i] for i in xu] , [v])
    # Stop if c1 is 0:
    if c1 == -1.0:
        return 'Error'
    else:
        # Sherman–Morrison formula:
        return (xb - (np.dot(c2,xb)/(1.0 + c1)))


A = [[1,2,-4],[-7,2,6],[3,1,2]]
b = [-7,-31,4]
lu = LU(A)
print LU_solve(lu,b)

# Part E
u = [1,0,0]
v = [0,2,0]
print Sherman_Morrison(u,v,b,lu)
# Part F
print LU_solve(LU(A + np.dot([[i] for i in u],[v])),b)

# Part G
u = [0,5,2]
v = [0,-1,0]
# (A + u v^T) is singular
print Sherman_Morrison(u,v,b,lu)

# Part H
u = [1,1,1]
v = [-1,-1,-1]
print Sherman_Morrison(u,v,b,lu)
