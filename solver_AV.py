import numpy as np
from scipy import linalg

import generatemats
from quadrules import quadrules
import matplotlib.pyplot as plt
import exactriemann


p = 3
dom = [0., 1.]
nt = 100
nplot = nt//4
nElems = 20
dt = 1e-5
gamma = 1.4
mu = 10
solptstype = 'gausslegendre'
fluxsplitmethod = ['centered', 'rusanov', 'guermond', 'roe', 'exact'][3]
case = ['sod', 'woodwardcolella', 'shu-osher'][0]
lbc = ['free', 'fixed'][1]
rbc = ['free', 'fixed'][1]




modalscale = False
shockvar = 0 # Density

modallimiter = False
maxmodalratio = 1./(p+1)**4 # Based on Persson & Peraire 2012


def initialConditions(sol, x):
	global gamma
	if case == 'sod' or case == 'woodwardcolella':
		(npts, _) = np.shape(sol)
		if case == 'sod':
			[rl, ul, pl] = [1., 0., 1.]
			[rr, ur, pr] = [0.125, 0., 0.1]
		elif case == 'woodwardcolella':
			[rl, ul, pl] = [1., 0., 1e3]
			[rr, ur, pr] = [1, 0., 1e-2]

		# Left state
		sol[:npts//2,0] = rl
		sol[:npts//2,1] = rl*ul
		# E = P/(gamma-1) + 0.5*rho*u^2
		sol[:npts//2,2] = pl/(gamma-1.) + 0.5*rl*ul**2

		# Right state
		sol[npts//2:,0] = rr
		sol[npts//2:,1] = rr*ur
		sol[npts//2:,2] = pr/(gamma-1.) + 0.5*rr*ur**2

		lval = np.array([rl, rl*ul, pl/(gamma-1.) + 0.5*rl*ul**2])
		rval = np.array([rr, rr*ur, pr/(gamma-1.) + 0.5*rr*ur**2])
	elif case == 'shu-osher':
		scale = (dom[1] - dom[0])/10.
		(npts, _) = np.shape(sol)
		splitidx = np.where(x>scale)[0][0]
		# Left state
		[rl, ul, pl] = [3.857143, 2.629369, 10.3333]
		sol[:splitidx,0] = rl
		sol[:splitidx,1] = rl*ul
		# P = 1.0,     E = P/(gamma-1) + 0.5*rho*u^2
		sol[:splitidx,2] = pl/(gamma-1.) + 0.5*rl*ul**2
		# Right state
		[rr, ur, pr] = [1. + 0.2*np.sin((1./scale)*(5*(x[splitidx:] - np.mean(x)))), 0., 1.]
		sol[splitidx:,0] = rr
		sol[splitidx:,1] = rr*ur
		# P = 0.125,     E = P/(gamma-1) + 0.5*rho*u^2
		sol[splitidx:,2] = pr/(gamma-1.) + 0.5*rr*ur**2

		lval = np.array([rl, rl*ul, pl/(gamma-1.) + 0.5*rl*ul**2])
		rval = np.array([1., rr*ur, pr/(gamma-1.) + 0.5*rr*ur**2])

	return [sol, lval, rval]

def divF(sol):
	global p, nElems, detJ, dt
	global interpLeft, interpRight, solGradMat, leftcorrGrad, rightcorrGrad, invLegVDM
	global pscale, divscale, modalscale, shockvar, modalratio, vsgraph, vsnadj, modallimiter, maxmodalratio
	global mu
	stride = p+1
	divF = np.zeros_like(sol)
	for eidx in range(nElems):
		locsol = sol[stride*eidx:stride*(eidx+1), :]
		tauxx = mu*calcGradU(locsol, eidx)
		if eidx == 0:
			tauxxleft = np.flip(tauxx, axis=0)
		else:
			leftsol = sol[stride*(eidx-1):stride*eidx, :]
			tauxxleft = mu*calcGradU(leftsol, eidx-1)
		if eidx == nElems - 1:
			tauxxright = np.flip(tauxx, axis=0)
		else:
			rightsol = sol[stride*(eidx+1):stride*(eidx+2), :]
			tauxxright = mu*calcGradU(rightsol, eidx+1)
		locf = calcFlux(locsol, tauxx)

		divF_uncorr = np.matmul(solGradMat,locf)
		dfl = commonFlux('left', eidx, tauxxleft, tauxx) - calcFlux(np.matmul(interpLeft,locsol), np.matmul(interpLeft,tauxx))
		dfr = commonFlux('right', eidx, tauxx, tauxxright) - calcFlux(np.matmul(interpRight,locsol), np.matmul(interpRight,tauxx))
		divF_corr = divF_uncorr + np.outer(leftcorrGrad, dfl) + np.outer(rightcorrGrad, dfr)

		rhs = divF_corr/detJ

		divF[stride * eidx:stride * (eidx + 1), :] = rhs

	return divF

def calcFlux(sol, tauxx):
	global gamma

	if np.ndim(sol) == 1:
		sol = np.reshape(sol,(1,len(sol)))

	f = np.zeros_like(sol)
	rho = sol[:,0]
	u = sol[:,1]/sol[:,0]
	E = sol[:,2]
	P = (gamma - 1.0)*(E - 0.5*rho*u**2)

	f[:,0] = rho*u
	f[:,1] = rho*u**2 + P - tauxx
	f[:,2] = u*(E + P - tauxx)

	return f

def calcGradU(sol, eidx):
	global detJ
	r  = sol[:,0]
	ru = sol[:,1]
	u = ru/r
	gradu_uncorr = (1./r)*(np.matmul(solGradMat, ru) - u*np.matmul(solGradMat, r))
	cs = commonSol('left', eidx)
	dul = 0.5*(cs[0][1]/cs[0][0] + cs[1][1]/cs[1][0]) - np.matmul(interpLeft,u)
	cs = commonSol('right', eidx)
	dur = 0.5*(cs[0][1]/cs[0][0] + cs[1][1]/cs[1][0]) - np.matmul(interpRight,u)

	gradu_corr =  gradu_uncorr + dul*leftcorrGrad + dur*rightcorrGrad
	return gradu_corr/detJ

def calcWaveSpeed(sol_left, sol_right, method):
	if method  == 'davis':
		rl = sol_left[0]
		ul = sol_left[1]/sol_left[0]
		El = sol_left[2]
		Pl = (gamma - 1.0)*(El - 0.5*rl*ul**2)

		rr = sol_right[0]
		ur = sol_right[1]/sol_right[0]
		Er = sol_right[2]
		Pr = (gamma - 1.0)*(Er - 0.5*rr*ur**2)

		cl = np.sqrt(gamma*Pl/rl)
		cr = np.sqrt(gamma*Pr/rr)
		cmax = max(np.abs(ul) + cl, np.abs(ur) + cr)
		return cmax
	if method  == 'guermond':
		rl = sol_left[0]
		ul = sol_left[1]/sol_left[0]
		El = sol_left[2]
		Pl = (gamma - 1.0)*(El - 0.5*rl*ul**2)
		cl = np.sqrt(gamma*Pl/rl)

		rr = sol_right[0]
		ur = sol_right[1]/sol_right[0]
		Er = sol_right[2]
		Pr = (gamma - 1.0)*(Er - 0.5*rr*ur**2)
		cr = np.sqrt(gamma*Pr/rr)

		pstar = ((cl + cr - 0.5*(gamma-1.)*(ur-ul))  /(   cl*(Pl**(-(gamma-1.)/(2.*gamma))) + cr*(Pr**(-(gamma-1.)/(2.*gamma)))  )    )**((2.*gamma)/(gamma-1.))
		laml = ul - cl*(1 + ((gamma+1.)/(2.*gamma))*(pstar - Pl)/Pl)**0.5
		lamr = ur + cr*(1 + ((gamma+1.)/(2.*gamma))*(pstar - Pr)/Pr)**0.5
		cmax = max(laml, lamr)
		return cmax
	if method  == 'exact':
		[r, P, u, e, cmax] = exactriemann.Solve(gamma, sol_left, sol_right)
		return cmax

def fluxStabTerm(sol_left, sol_right, method):
	global gamma
	if method  == 'centered':
		return 0.
	elif method  == 'rusanov' or method == 'davis':
		cmax = calcWaveSpeed(sol_left, sol_right, 'davis')
		return cmax*(sol_left - sol_right)
	elif method  == 'guermond':
		cmax = calcWaveSpeed(sol_left, sol_right, 'guermond')
		return cmax*(sol_left - sol_right)
	elif method == 'roe':
		[rl, rul, El] = sol_left
		[rr, rur, Er] = sol_right
		ul = rul/rl
		ur = rur/rr
		Pl = (gamma - 1.0)*(El - 0.5*rl*ul**2)
		Pr = (gamma - 1.0)*(Er - 0.5*rr*ur**2)
		hl = El + Pl/rl
		hr = Er + Pr/rr

		rRL = np.sqrt(rl*rr)
		uRL = (np.sqrt(rl)*ul + np.sqrt(rr)*ur)/(np.sqrt(rl) + np.sqrt(rr))
		hRL = (np.sqrt(rl)*hl + np.sqrt(rr)*hr)/(np.sqrt(rl) + np.sqrt(rr))
		aRL = np.sqrt((gamma-1.)*(hRL - 0.5*uRL**2))

		[lam1, lam2, lam3] = [np.abs(uRL), np.abs(uRL + aRL), np.abs(uRL - aRL)]

		# Entropy fixe
		eps = np.abs(ur - ul)
		if lam1 < eps:
			lam1 = (0.5/eps)*(lam1**2 + eps**2)
		if lam3 < eps:
			lam3 = (0.5/eps)*(lam3**2 + eps**2)

		eigRL1 = np.array([1., uRL, 0.5*uRL**2])
		eigRL2 = (rRL/(2.*aRL))*np.array([1., uRL + aRL, hRL + uRL*aRL])
		eigRL3 = -(rRL/(2.*aRL))*np.array([1., uRL - aRL, hRL - uRL*aRL])

		dw1 = (rr - rl) - (Pr - Pl)/(aRL**2)
		dw2 = (ur - ul) + (Pr - Pl)/(rRL*aRL)
		dw3 = (ur - ul) - (Pr - Pl)/(rRL*aRL)

		df = eigRL1*lam1*dw1 + eigRL2*lam2*dw2 + eigRL3*lam3*dw3
		return -df

def fluxSplit(sol_left, sol_right, tauxxint):
	global fluxsplitmethod, gamma

	if fluxsplitmethod  == 'exact':
		[r, P, u, e, cmax] = exactriemann.Solve(gamma, sol_left, sol_right)
		sol_exact = [r, r*u, P/(gamma-1.) + 0.5*r*u**2]
		return calcFlux(sol_exact, tauxxint)
	else:
		return (0.5*calcFlux(sol_left, tauxxint) + 0.5*calcFlux(sol_right, tauxxint) + 0.5*fluxStabTerm(sol_left, sol_right, fluxsplitmethod))

def commonSol(side, eidx):
	global sol, p
	global interpLeft, interpRight
	global lbc, rbc, lval, rval

	stride = p+1
	if side == 'left':
		# Get solution at basis points
		sol_right = sol[eidx    *stride:(eidx+1)*stride, :]
		sol_rint = np.matmul(interpLeft, sol_right)
		if eidx == 0:
			if lbc == 'fixed':
				sol_lint = lval
			else:
				sol_left = np.flip(sol_right, axis=0)
				sol_lint = np.matmul(interpRight, sol_left)
		else:
			sol_left = sol[(eidx-1)*stride:eidx    *stride, :]
			sol_lint = np.matmul(interpRight, sol_left)
	else:
		# Get solution at basis points
		sol_left = sol[eidx*stride    :(eidx+1)*stride, :]
		sol_lint = np.matmul(interpRight, sol_left)
		if eidx == nElems - 1:
			if rbc == 'fixed':
				sol_rint = rval
			else:
				sol_right = np.flip(sol_left, axis=0)
				sol_rint = np.matmul(interpLeft, sol_right)
		else:
			sol_right = sol[(eidx+1)*stride:(eidx+2)*stride, :]
			sol_rint = np.matmul(interpLeft, sol_right)

	return [sol_lint, sol_rint]

def commonFlux(side, eidx, tauxxleft, tauxxright):
	[sol_lint, sol_rint] = commonSol(side, eidx)
	tauxxint = 0.5*np.matmul(interpRight, tauxxleft) + 0.5*np.matmul(interpLeft, tauxxright)
	fcomm = fluxSplit(sol_lint, sol_rint, tauxxint)
	return fcomm

def step(sol,dt):
	k1 = -dt*divF(sol)
	k2 = -dt*divF(sol+k1/2.)
	k3 = -dt*divF(sol+k2/2.)
	k4 = -dt*divF(sol+k3)
	return sol + k1/6. + k2/3. + k3/3. + k4/6.

def getPointLocations(basispts, dom, nElems):
	ptsperelem = len(basispts)
	x = np.zeros(ptsperelem*nElems)
	for i in range(nElems):
		xi = dom[0] + i*(dom[1] - dom[0])/(nElems)
		xf = dom[0] + (i+1)*(dom[1] - dom[0])/(nElems)
		for j in range(ptsperelem):
			x[i*ptsperelem + j] = xi + (xf-xi)*0.5*(basispts[j]+1.)
	return x

def integrateDomain(sol):
	global basiswts
	global nElems
	global p

	domint = np.zeros(3)
	stride = p+1
	for i in range(nElems):
		domint[:] += np.dot(basiswts, sol[i*stride:(i+1)*stride,:])
	return domint/len(sol)



detJ = (dom[1] - dom[0])/(2.0*nElems)
nvars = 3
sol = np.ones((nElems*(p+1), nvars))

# Get basis points in computational space
[basispts, basiswts] = quadrules.getPtsAndWeights(p, solptstype) 
x = getPointLocations(basispts, dom, nElems)
[sol, lval, rval] = initialConditions(sol, x)

# Get interpolation and differentiation matrices
interpLeft = generatemats.interpVect(basispts, -1.)
interpRight = generatemats.interpVect(basispts, 1.)
solGradMat = generatemats.solGradMat(basispts)
leftsolGrad = generatemats.solGradInterVect(basispts, -1.)
rightsolGrad = generatemats.solGradInterVect(basispts, 1.)

leftcorrGrad = generatemats.corrGradVect(basispts, 'left')
rightcorrGrad = generatemats.corrGradVect(basispts, 'right')
invLegVDM = np.linalg.inv(np.polynomial.legendre.legvander(basispts, p))


for t in range(nt):
	sol = step(sol, dt)
	#print('T: ', t*dt, 'L_inf norm: ', np.max(sol, axis=0))
	#print('T: ', t*dt, 'L_2 norm: ', np.linalg.norm(sol, axis=0))
	print('T: ', t*dt, 'Integral: ', integrateDomain(sol))
	if t%nplot == 0 or t == nt-1:
		plt.figure(0)
		plt.plot(x, sol[:,0], '.-')
		plt.xlabel('x')
		plt.ylabel('Density')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, mu = {4}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, mu))
		plt.figure(1)
		plt.plot(x, sol[:,1], '.-')
		plt.xlabel('x')
		plt.ylabel('Velocity')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, mu = {4}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, mu))
		plt.figure(2)
		plt.plot(x, (gamma - 1.0)*(sol[:,2] - 0.5*(sol[:,1]**2)/sol[:,0]) , '.-')
		plt.xlabel('x')
		plt.ylabel('Pressure')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, mu = {4}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, mu))

plt.show()




