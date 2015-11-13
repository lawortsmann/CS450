# -*- coding: utf-8 -*-
# HW6.py
"""
CS 450 Homework 6
@version: 11.09.2015
@author: wortsma2
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sci_interp

# Problem 1:

class interpolation:
"""
Class to interpolate a function with n piviot points.
points can be uniform or chebyshev.
interpolation can be Lagrange polynomial or cubic spline.
cubic spline interpolation uses scipy.interpolate.splXX

get_residual method calculates the norm of the residual of the interpolation
    using n sample points of the function/interpolation
"""
    def __init__(self, f, n, lb, ub, sampling = 'uniform'):
        self.lb = lb
        self.ub = ub
        self.n = n
        self.f = f
        self.sampling = sampling
        if sampling == 'uniform':
            self.pts = self.get_uniform_points(f, n, lb, ub)

        elif sampling == 'chebyshev':
            self.pts = self.get_chebyshev_points(f, n, lb, ub)

        else:
            self.pts = self.get_uniform_points(f, n, lb, ub)

    def get_uniform_points(self, f, n, lb, ub):
        pts_x = np.linspace(lb, ub, n)
        pts_y = np.array([f(x) for x in pts_x])
        return pts_x, pts_y

    def get_chebyshev_points(self, f, n, lb, ub):
        pts_x = np.array([np.cos(((2.*(i+1) - 1.)/(2.*n))*np.pi) for i in xrange(n)])
        pts_y = np.array([f(x) for x in pts_x])
        return pts_x, pts_y

    def evaluate_lagrange(self, x):
        poly = 0.0
        for i in xrange(self.n):
            degree = self.pts[1][i]
            for j in xrange(self.n):
                if i != j:
                    degree *= (x - self.pts[0][j])/(self.pts[0][i] - self.pts[0][j])
            poly += degree
        return poly

    def get_spline(self):
        self.spline = sci_interp.splrep(self.pts[0], self.pts[1], s = 0)
        return self.spline

    def get_residual(self, n, interp = 'lagrange'):
        if self.sampling == 'uniform':
            x,yr = self.get_uniform_points(self.f, n, self.lb, self.ub)

        elif self.sampling == 'chebyshev':
            x,yr = self.get_chebyshev_points(self.f, n, self.lb, self.ub)

        else:
            x,yr = self.get_uniform_points(self.f, n, self.lb, self.ub)

        if interp == 'lagrange':
            yi = np.array([self.evaluate_lagrange(i) for i in x])

        elif interp == 'spline':
            self.get_spline()
            yi = sci_interp.splev(x, self.spline, der = 0)

        else:
            yi = np.array([self.evaluate_lagrange(i) for i in x])

        return np.linalg.norm(yi-yr)

def f1(x):
    return 1./(1. + x*x*25.)

def f2(x):
    return np.exp(np.cos(x))

# find all residuals for f1 and plot:
ns = range(4,51)
resids_chebyshev = []
resids_uniform = []
resids_spline = []
for n in ns:
    test1 = interpolation(f1, n, -1, 1, sampling = 'chebyshev')
    test2 = interpolation(f1, n, -1, 1, sampling = 'uniform')
    resids_chebyshev.append(test1.get_residual(10*n, interp = 'lagrange'))
    resids_uniform.append(test2.get_residual(10*n, interp = 'lagrange'))
    resids_spline.append(test2.get_residual(10*n, interp = 'spline'))

plt.semilogy(ns, resids_chebyshev, label = 'chebyshev')
plt.semilogy(ns, resids_uniform, label = 'uniform')
plt.semilogy(ns, resids_spline, label = 'spline')
plt.title('Error of interpolation of f1')
plt.legend()
plt.show()

# And for f2:
ns = range(4,51)
resids_chebyshev = []
resids_uniform = []
resids_spline = []
for n in ns:
    test1 = interpolation(f2, n, 0, 2*np.pi, sampling = 'chebyshev')
    test2 = interpolation(f2, n, 0, 2*np.pi, sampling = 'uniform')
    resids_chebyshev.append(test1.get_residual(10*n, interp = 'lagrange'))
    resids_uniform.append(test2.get_residual(10*n, interp = 'lagrange'))
    resids_spline.append(test2.get_residual(10*n, interp = 'spline'))

plt.semilogy(ns, resids_chebyshev, label = 'chebyshev')
plt.semilogy(ns, resids_uniform, label = 'uniform')
plt.semilogy(ns, resids_spline, label = 'spline')
plt.title('Error of interpolation of f2')
plt.legend()
plt.show()

# Problem 3:

class SplineBuilder:
    """
    Class to allow interaction with matplotlib.
    Draws spline ontop of image, knot points chosen by clicking
    get_knot_points adjusts and returns the knot points
    """
    def __init__(self, line):
        self.line = line
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        if len(self.xs) > 3:
            tck,uk = sci_interp.splprep([self.xs,self.ys], s = 0)
            self.uk = uk
            u = np.linspace(0, 1, 1000)
            out = sci_interp.splev(u, tck)
            self.line.set_data(out[0], out[1])
        else:
            self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def get_knot_points(self):
        nx = np.array(self.xs)
        ny = -1.0*np.array(self.ys)
        sx = min(nx)
        lx = max(nx)
        sy = min(ny)
        rx = (lx-sx)
        ny = (ny-sy)/rx
        nx = (nx-sx)/rx
        return self.uk,nx,ny

# First let the user draw the spline by tracing the letter:
img = plt.imread('L.png')
fig = plt.figure()
plt.imshow(img,cmap='gray')
ax = fig.add_subplot(111)
line, = ax.plot([0],[0], linewidth = 10)
spline = SplineBuilder(line)
plt.show()
# Now just plot the knots/spline:
kpt,kpx,kpy = spline.get_knot_points()
tck,ui = sci_interp.splprep([kpx,kpy], s = 0)
u = np.linspace(0, 1, 1000)
nx,ny = sci_interp.splev(u, tck)
plt.plot(kpx, kpy,'ro')
plt.plot(nx, ny)
plt.show()
# Print knot points:
print np.array([kpt,kpx,kpy]).T

# Problem 5:

def pi_f(x):
    return 4./(1. + x**2.)

def odd(x):
    """
    return an odd int near x
    """
    if int(x)%2 == 1:
        return int(x)
    else:
        return int(x)+1

def trapezoidal(N):
    """
    trapezoidal rule for N points
    """
    h = 1./(N-1)
    quad = 0.
    for i in xrange(N-1):
        x1 = i*h
        x2 = (i+1)*h
        quad += (h/2.)*(pi_f(x1) + pi_f(x2))
    return abs(np.pi - quad)

def midpoint(N):
    """
    midpoint rule for N points
    """
    h = 1./(N-1.)
    quad = 0.
    for i in xrange(N-1):
        x = (h/2.) + i*h
        quad += h*pi_f(x)
    return abs(np.pi - quad)

def simpson(N):
    """
    simpson's rule for N points
    """
    h = 1./(N - 1.)
    quad = 0.0
    for i in xrange(N):
        fi = pi_f(i*h)
        if (i == 0) or i == (N-1):
            si = 1.
        elif i%2 == 1:
            si = 4.
        else:
            si = 2.
        quad += (h/3.)*fi*si
    return abs(np.pi - quad)

# Calculate errors:
ns = map(odd,np.logspace(1,5,55))
hvals = []
error_trap = []
error_simp = []
error_mid = []
for n in ns:
    hvals.append(1./float(n-1))
    error_trap.append(trapezoidal(n))
    error_simp.append(simpson(n))
    error_mid.append(midpoint(n))

# Plot errors:
plt.loglog(hvals,error_trap, label = 'Trapezoidal Rule')
plt.loglog(hvals,error_simp, label = 'Simpson\'s Rule')
plt.loglog(hvals,error_mid, label = 'Midpoint Rule')
plt.legend(loc=2)
plt.xlabel('h')
plt.ylabel('error')
plt.show()
