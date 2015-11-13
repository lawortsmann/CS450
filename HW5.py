# -*- coding: utf-8 -*-
# HW5.py
"""
@version: 10.28.2015
@author: wortsma2
"""
import numpy as np
import matplotlib.pyplot as plt

# Problem 2:

def f(x):
    x1 = x[0]
    x2 = x[1]
    f1 = (x1 + 3.)*(x2**3. - 7.) + 18.
    f2 = np.sin(x2*np.exp(x1) - 1.)
    return np.array([f1,f2])

def jacobi_f(x):
    x1 = x[0]
    x2 = x[1]
    j11 = x2**3. - 7.
    j12 = 3.*(3. + x1)*x2**2.
    j21 = np.exp(x1)*x2*np.cos(1. - np.exp(x1)*x2)
    j22 = np.exp(x1)*np.cos(1. - np.exp(x1)*x2)
    return np.array([[j11,j12],[j21,j22]])

print 'Newtons\'s method:'

x = np.array([-0.5, 1.4])
MAX_ITERATIONS = 25
TOL = 1e-16
e_n = []
for i in xrange(MAX_ITERATIONS):
    fxk = f(x)
    e_n.append(np.linalg.norm(x - np.array([0,1])))
    if np.linalg.norm(fxk) < TOL:
        break
    x += np.linalg.solve(jacobi_f(x),-fxk)
if i < 24:
    print 'Solution: ' + str(x.round(3))
    print 'Converged in '+str(i)+' iterations'
else:
    print 'Solution did not converge in '+str(MAX_ITERATIONS)+' iterations'
print
print 'Broyden\'s method:'

x = np.array([-0.5, 1.4])
B = jacobi_f(x)
MAX_ITERATIONS = 25
TOL = 1e-16
e_b = []
for i in xrange(MAX_ITERATIONS):
    fxk = f(x)
    e_b.append(np.linalg.norm(x - np.array([0,1])))
    if np.linalg.norm(fxk) < TOL:
        break
    s = np.linalg.solve(B,-fxk)
    x += s
    y = np.array([f(x) - fxk]).T
    s = np.array([s]).T
    div = (s.T).dot(s)
    num = (y - B.dot(s)).dot(s.T)
    B += num/div
if i+1 < MAX_ITERATIONS:
    print 'Solution: ' + str(x.round(3))
    print 'Converged in '+str(i)+' iterations'
else:
    print 'Solution did not converge in '+str(MAX_ITERATIONS)+' iterations'

plt.semilogy(e_n)
plt.title('Convergence of Newtons\'s method')
plt.show()
plt.semilogy(e_b)
plt.title('Convergence of Broyden\'s method')
plt.show()


# Problem 4:

def f(x):
    x1 = x[0,0]
    x2 = x[1,0]
    return 100.*((x2 - x1**2.)**2.) + ((1. - x1)**2.)

def grad_f(x):
    x1 = x[0]
    x2 = x[1]
    g1 = 2.*(x1 + 200.*x1**3. - 200.*x1*x2 - 1.)
    g2 = 200.*(x2 - x1**2.)
    return np.array([g1,g2])

def hessian_f(x):
    x1 = x[0]
    x2 = x[1]
    h11 = 2. + 800.*x1**2. - 400.*(x2 - x1**2.)
    h12 = -400.*x1
    h21 = -400.*x1
    h22 = 200.
    return np.array([[h11,h12],[h21,h22]])

def line_search(a, x):
    return f(x - a*grad_f(x))

def secant(f, x0, maxiterations = 200, tol = 1e-15):
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
    return xk,i

def steepest(x0, MAX_ITERATIONS = 50, TOL = 1e-15):
    x = x0
    for i in xrange(MAX_ITERATIONS):
        a,j = secant(lambda a: line_search(a, x), 1)
        if a < TOL:
            break
        x = x - a*grad_f(x)

    if i+1 < MAX_ITERATIONS:
        print 'Solution: ' + str(x[:,0].round(3))
        print 'f(x) = '+str(f(x))
        print 'Stopped in '+str(i)+' iterations'
    else:
        print 'Solution did not converge in '+str(MAX_ITERATIONS)+' iterations'

def newton_opt(x0, MAX_ITERATIONS = 50, TOL = 1e-15):
    x = x0
    for i in xrange(MAX_ITERATIONS):
        s = np.linalg.solve(hessian_f(x), -grad_f(x))
        if np.linalg.norm(s) < TOL:
            break
        x = x + s
    if i+1 < MAX_ITERATIONS:
        print 'Solution: ' + str(x[:,0].round(3))
        print 'f(x) = '+str(f(x))
        print 'Stopped in '+str(i)+' iterations'
    else:
        print 'Solution did not converge in '+str(MAX_ITERATIONS)+' iterations'

print 'Actual minimum at x = [1,1]'
print 'f(x) = 0'
print

x1 = np.array([[-1],[1]])
print 'Steepest Descent for x0 = '+str(x1[:,0])+':'
steepest(x1)
print
print 'Newton\'s Method for x0 = '+str(x1[:,0])+':'
newton_opt(x1)
print

x2 = np.array([[0],[1]])
print 'Steepest Descent for x0 = '+str(x2[:,0])+':'
steepest(x2)
print
print 'Newton\'s Method for x0 = '+str(x2[:,0])+':'
newton_opt(x2)
print

x3 = np.array([[2],[1]])
print 'Steepest Descent for x0 = '+str(x3[:,0])+':'
steepest(x3)
print
print 'Newton\'s Method for x0 = '+str(x3[:,0])+':'
newton_opt(x3)
print

# Problem 5:

def f(x):
    return 0.5*(x**2.) - np.sin(x)

def golden_section(a, b, MAX_ITERATIONS = 100, TOL = 1e-15):
    t = 0.5*(5.**0.5 - 1.)
    x1 = a + (1.-t)*(b-a)
    x2 = a + t*(b-a)
    f1 = f(x1)
    f2 = f(x2)
    for i in xrange(MAX_ITERATIONS):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + t*(b-a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1.-t)*(b-a)
            f1 = f(x1)
        if (b-a) < TOL:
            break

    x = 0.5*(x1+x2)
    if i+1 < MAX_ITERATIONS:
        print 'Solution: x = ' + str(x)
        print 'f(x) = '+str(f(x))
        print 'Stopped in '+str(i)+' iterations'
    else:
        print 'Search did not stop in '+str(MAX_ITERATIONS)+' iterations'

golden_section(0,4)
