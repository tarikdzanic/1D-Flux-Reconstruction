import numpy as np
from scipy import linalg

import generatemats
from quadrules import quadrules
import matplotlib.pyplot as plt


# --------- PARAMETERS ------------



p = 1
dom = [0., 1.]
nt = 200
nplot = 20
nElems = 100
dt = 1e-4
solptstype = 'gausslegendre'
fluxsplitmethod = 'rusanov'
gamma = 1.4


def initialConditions(sol):
	global gamma
	(npts, nvars) = np.shape(sol)
	# Left state
	[rl, ul, pl] = [1., 0., 1.]
	sol[:npts//2,0] = 1.0
	sol[:npts//2,1] = 0.0
	# P = 1.0,     E = P/(gamma-1) + 0.5*rho*u^2
	sol[:npts//2,2] = pl/(gamma-1.) + 0.5*rl*ul**2
	# Right state
	[rr, ur, pr] = [0.125, 0., 0.1]
	sol[npts//2:,0] = 0.125
	sol[npts//2:,1] = 0.0
	# P = 0.125,     E = P/(gamma-1) + 0.5*rho*u^2
	sol[npts//2:,2] = pr/(gamma-1.) + 0.5*rr*ur**2
	return sol


def divF(sol):
	global nElems
	global nvars
	global p
	global interpLeft
	global interpRight
	global solGradMat 
	global leftcorrGrad 
	global rightcorrGrad 
	global J

	stride = p+1
	divF = np.zeros_like(sol)
	for eidx in range(nElems):
		locsol = sol[stride*eidx:stride*(eidx+1), :]
		locf = calcFlux(locsol)

		divF_uncorr = np.matmul(solGradMat,locf)
		dfl = commonFlux('left', eidx) - calcFlux(np.matmul(interpLeft,locsol))
		dfr = commonFlux('right', eidx) - calcFlux(np.matmul(interpRight,locsol))
		divF_corr = divF_uncorr + np.outer(leftcorrGrad, dfl) + np.outer(rightcorrGrad, dfr)

		divF[stride*eidx:stride*(eidx+1),:] = divF_corr*J
	return divF

def calcFlux(sol):
	global gamma

	if np.ndim(sol) == 1:
		sol = np.reshape(sol,(1,len(sol)))

	f = np.zeros_like(sol)
	rho = sol[:,0]
	u = sol[:,1]/sol[:,0]
	E = sol[:,2]
	P = (gamma - 1.0)*(E - 0.5*rho*u**2) 

	f[:,0] = rho*u
	f[:,1] = rho*u**2 + P
	f[:,2] = u*(E + p)

	return f

def fluxSplit(sol_left, sol_right):
	global fluxsplitmethod 
	global gamma
	if fluxsplitmethod  == 'centered':
		return (0.5*calcFlux(sol_left) + 0.5*calcFlux(sol_right))
	elif fluxsplitmethod  == 'rusanov':
		rl = sol_left[0]
		ul = sol_left[1]/sol_left[0]
		El = sol_left[2]
		Pl = (gamma - 1.0)*(El - 0.5*rl*ul**2) 

		rr = sol_right[0]
		ur = sol_right[1]/sol_right[0]
		Er = sol_right[2]
		Pr = (gamma - 1.0)*(Er - 0.5*rr*ur**2)

		amax = np.sqrt(0.25*gamma*(Pl + Pr)/(rl + rr)) + 0.25*np.abs(ul + ur)
		return (0.5*calcFlux(sol_left) + 0.5*calcFlux(sol_right) - amax*(sol_left - sol_right))

def commonFlux(side, eidx):
	global sol
	global p
	global interpLeft
	global interpRight
	global nvars

	stride = p+1
	if side == 'left':
		# Get solution at basis points
		sol_right = sol[eidx    *stride:(eidx+1)*stride, :]
		if eidx == 0:
			sol_left = np.flip(sol_right, axis=0)
		else:
			sol_left  = sol[(eidx-1)*stride:eidx    *stride, :]
	else:
		# Get solution at basis points
		sol_left  = sol[eidx*stride    :(eidx+1)*stride, :]
		if eidx == nElems - 1:
			sol_right = np.flip(sol_left, axis=0)
		else:
			sol_right = sol[(eidx+1)*stride:(eidx+2)*stride, :]

	# Interpolate to interface
	sol_lint = np.matmul(interpRight, sol_left)
	sol_rint = np.matmul(interpLeft, sol_right)
	fcomm = fluxSplit(sol_lint, sol_rint)

	return fcomm

def step(sol,dt):
	k1 = dt*divF(sol)
	k2 = dt*divF(sol+k1/2.)
	k3 = dt*divF(sol+k2/2.)
	k4 = dt*divF(sol+k3)
	return sol + k1/6. + k2/3. + k3/3. + k4/6.


J = 2.0*nElems/(dom[1] - dom[0])
nvars = 3
sol = np.ones((nElems*(p+1), nvars))
sol = initialConditions(sol)

# Get basis points in computational space
[basispts, basiswts] = quadrules.getPtsAndWeights(p, solptstype) 

# Get interpolation and differentiation matrices
interpLeft = generatemats.interpVect(basispts, -1.)
interpRight = generatemats.interpVect(basispts, 1.)
solGradMat = generatemats.solGradMat(basispts)
leftcorrGrad = generatemats.corrGradVect(basispts, 'left')
rightcorrGrad = generatemats.corrGradVect(basispts, 'right')

plt.figure()
for t in range(nt):
	sol = step(sol, dt)
	print('L_inf norm: ', np.max(sol, axis=0))
	if t%nplot == 0:
		plt.plot(sol[:,0])

plt.show()




