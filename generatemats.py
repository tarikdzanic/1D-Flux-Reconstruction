import numpy as np
from scipy import linalg
from scipy import interpolate


# M*u = du/dxi at pts xi=xi_i
def solGradMat(pts):
	npts = len(pts)
	M = np.zeros((npts, npts))

	for j in range(npts):
		vals = np.zeros(npts)
		vals[j] = 1.
		lag = interpolate.lagrange(pts, vals)
		dlag = lag.deriv()
		for i in range(npts):
			M[i,j] = dlag(pts[i])
	return M

# Returns dh/dxi(xi) for a correction function 
def corrGradVect(pts, side):
	npts = len(pts)
	p = npts - 1
	c = np.zeros(p+2)

	if side == 'left':
		c[p] = 0.5*(-1)**(p)
		c[p+1] = 0.5*(-1)**(p+1)
	else:
		c[p] = 0.5
		c[p+1] = 0.5

	L = np.polynomial.legendre.Legendre(c)
	Ld = L.deriv()

	M = np.zeros(npts)

	for i in range(npts):
		M[i] = Ld(pts[i])
	return M

def interpVect(pts, val):
	npts = len(pts)
	M = np.zeros(npts)

	for i in range(npts):
		vals = np.zeros(npts)
		vals[i] = 1.
		lag = interpolate.lagrange(pts, vals)
		M[i] = lag(val)
	return M


