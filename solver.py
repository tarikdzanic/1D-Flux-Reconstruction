import numpy as np
from scipy import linalg

import generatemats
from quadrules import quadrules
import matplotlib.pyplot as plt
import exactriemann
import pickle


p = 3
nt = 1000
nplot = nt//4
nElems = 128
dt = 2e-4
gamma = 1.4
solptstype = 'gausslegendre'
fluxsplitmethod = ['centered', 'rusanov', 'guermond', 'roe', 'exact'][2]
lambdamethod = ['davis', 'guermond', 'roe', 'exact'][1]
case = ['sod', 'woodwardcolella', 'shu-osher'][0]
lbc = ['free', 'fixed'][1]
rbc = ['free', 'fixed'][0]
timescheme = ['rk4', 'ssprk3'][1]

dom = [0., 10.] if case == 'shu-osher' else [0., 1.]


graphvisc = True
graphupwind = False

pscale = True
divscale = True
divnorm = 2

modalscale = False
shockvar = 0 # Density
modalratio = [np.log10(1./(p+1)**4), np.log10(100./(p+1)**4)] # Log10 of Modal ratio to start adding AV, modal ratio to add full AV

vsgraph = False
vsnadj = 1 # If == 0 and vsgraph,modalscale == True, determines nadj from modal coefficient raito

modallimiter = False
maxmodalratio = 1./(p+1)**4 # Based on Persson & Peraire 2012

outpath = 'C:\\Users\\Tarik-Personal\\Documents\\Research\\ShockCapturing\\data\\{0}_p{1}_dof{2}_l{3}_FS{4}_LM{5}_GVU{6}_DS{7}'.format(case, p, (p+1)*nElems, divnorm, fluxsplitmethod, lambdamethod, graphupwind, divscale)

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

	stride = p+1
	divF = np.zeros_like(sol)
	for eidx in range(nElems):
		locsol = sol[stride*eidx:stride*(eidx+1), :]
		locf = calcFlux(locsol)

		divF_uncorr = np.matmul(solGradMat,locf)
		dfl = commonFlux('left', eidx) - calcFlux(np.matmul(interpLeft,locsol))
		dfr = commonFlux('right', eidx) - calcFlux(np.matmul(interpRight,locsol))
		divF_corr = divF_uncorr + np.outer(leftcorrGrad, dfl) + np.outer(rightcorrGrad, dfr)
		dfints = [commonFlux('right', eidx)/(detJ), commonFlux('left', eidx)/(detJ)]
		#print(divF_corr, dfp0)

		rhs = divF_corr/detJ

		# If adding graph viscosity
		if graphvisc:
			# If scaling full graph viscosity by a sinusoidal blending factor or setting nadj by the blending factor
			if modalscale and p != 0:
				leg_coeffs = np.matmul(invLegVDM, locsol[:,shockvar])
				mag_ratio = np.linalg.norm(leg_coeffs[-1])/(np.linalg.norm(leg_coeffs) + 1e-12)

				if mag_ratio > modalratio[0]:
					if mag_ratio > modalratio[1]:
						blendfac = 1.0
					else:
						blendfac = 0.5*np.sin(np.pi*(mag_ratio - modalratio[0])/(modalratio[1] - modalratio[0]) - np.pi/2.0) + 0.5

					# If setting nadj by the blending factor
					if vsgraph:
						if vsnadj == 0:
							nadj = int(round(blendfac*(p+1)))
							if nadj == 0:
								graphsource = np.zeros_like(rhs)
							else:
								graphsource = getVSGraphSource(eidx, nadj)
						else:
							graphsource = getVSGraphSource(eidx, vsnadj)
					# Else set graph viscosity by multiplying blending factor by the full graph viscosity
					else:
						graphsource = blendfac*getGraphSource(eidx)
				else:
					graphsource = np.zeros_like(rhs)

			elif modallimiter and p != 0:
				leg_coeffs = np.matmul(invLegVDM, locsol[:, shockvar])
				mag_ratio = np.linalg.norm(leg_coeffs[-1])/(np.linalg.norm(leg_coeffs) + 1e-12)
				if mag_ratio < maxmodalratio:
					graphsource = np.zeros_like(rhs)
				else:
					for nadj in range(p+1):
						graphsource = getVSGraphSource(eidx, nadj)
						rhs_pred = normalizeGV(divF_corr/detJ, graphsource)
						u_pred = locsol - dt*rhs_pred
						leg_coeffs_pred = np.matmul(invLegVDM, u_pred[:, shockvar])
						mag_ratio_pred = np.linalg.norm(leg_coeffs_pred[-1])/(np.linalg.norm(leg_coeffs_pred) + 1e-12)
						if mag_ratio_pred < maxmodalratio:
							break

			elif vsgraph:
				graphsource = getVSGraphSource(eidx, vsnadj)
			else:
				graphsource = getGraphSource(eidx)

			if divscale:
				rhs = normalizeGV(divF_corr/detJ, graphsource, dfints)
			else:
				rhs += -graphsource
		divF[stride * eidx:stride * (eidx + 1), :] = rhs

	return divF

def normalizeGV(divF, graphsource, dfints):
	global basiswts
	global p
	# divF = divF_corr/detJ
	if divnorm == 1:
		divMag = np.dot(basiswts, np.abs(divF))
		gsMag = np.dot(basiswts, np.abs(graphsource))
	elif divnorm == 2:
		divMag = np.sqrt(np.dot(basiswts, divF**2))
		gsMag = np.sqrt(np.dot(basiswts, graphsource**2))
	elif divnorm == 'inf':
		divMag = np.amax(np.abs(divF), axis=0)
		gsMag = np.amax(np.abs(graphsource), axis=0)
	#divMag = np.linalg.norm(divF, axis=0)
	#gsMag = np.linalg.norm(graphsource, axis=0)
	unscaledSum = divF - graphsource

	# L2
	normfac = divMag/(divMag + gsMag + 1e-12)
	rhs = unscaledSum*normfac

	# P0
	#dfp0 = dfints[0] + (dfints[1] - dfints[0])*0.5*(basispts + 1.0)
	#normfac = np.abs(dfp0)/(divMag + gsMag + 1e-12)
	#rhs = unscaledSum*normfac
	return rhs

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
	f[:,2] = u*(E + P)

	return f

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

def fluxSplit(sol_left, sol_right):
	global fluxsplitmethod, gamma

	if fluxsplitmethod  == 'exact':
		[r, P, u, e, cmax] = exactriemann.Solve(gamma, sol_left, sol_right)
		sol_exact = [r, r*u, P/(gamma-1.) + 0.5*r*u**2]
		return calcFlux(sol_exact)
	else:
		return (0.5*calcFlux(sol_left) + 0.5*calcFlux(sol_right) + 0.5*fluxStabTerm(sol_left, sol_right, fluxsplitmethod))

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

def commonFlux(side, eidx):
	[sol_lint, sol_rint] = commonSol(side, eidx)
	fcomm = fluxSplit(sol_lint, sol_rint)
	return fcomm

def interpFluxSplit(ul, ur):
	tol = 1e-5
	f = [calcFlux(ul)[0],calcFlux(ur)[0], fluxSplit(ul, ur)[0]]

	interps = np.zeros((nvars, 2))
	for j in range(nvars):
		if abs(f[2][j]-f[0][j]) < tol and abs(f[2][j]-f[1][j]) < tol:
			interps[j, :] = [0.5, 0.5]
		elif abs(f[2][j]-f[0][j]) < tol:
			interps[j, :] = [1., 0.]
		elif abs(f[2][j]-f[1][j]) < tol:
			interps[j, :] = [0., 1.]
		else:
			fint = (f[2][j] - f[0][j])/(f[1][j] - f[0][j] + tol)
			fint = min(max(0., fint), 1.)
			interps[j, :] = [ 1.-fint,  fint]
	return interps

def getGraphSource(eidx):
	global sol, nElems, p, detJ
	global solGradMat, leftcorrGrad, rightcorrGrad, interpLeft, interpRight, leftsolGrad, rightsolGrad
	global lambdamethod, basiswts, graphupwind

	stride = p+1
	u = sol[stride*eidx:stride*(eidx+1),:]
	graphsource = np.zeros_like(u)

	[ul_ext, ul_int]  = commonSol('left', eidx)
	[ur_int, ur_ext]  = commonSol('right', eidx)

	# u at left/right flux points
	ufl = np.vstack((ul_ext, ul_int))
	ufr = np.vstack((ur_int, ur_ext))

	if graphupwind:
		nl = interpFluxSplit(ul_ext, ul_int)
		nr = -interpFluxSplit(ur_int, ur_ext)
	else:
		nl = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
		nr = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])


	for i in range(p+1):
		for j in range(p+1):
			# SOLUTION POINTS
			cij = max(np.abs(solGradMat[i,j]), np.abs(solGradMat[j,i]))
			#cij = np.abs(solGradMat[i,j])
			dij = cij*fluxStabTerm(u[j,:], u[i,:], lambdamethod)/detJ
			if pscale:
				graphsource[i,:] += dij*basiswts[i]/(p+1)
			else:
				graphsource[i,:] += dij

		# Left flux points
		for j in range(2):
			#cij = max(0.5*np.abs(leftcorrGrad[i]), np.abs(leftsolGrad[i]))
			cij = np.abs(nl[:,j]*leftcorrGrad[i])
			dij = cij*fluxStabTerm(ufl[j,:], u[i,:], lambdamethod)/detJ
			if pscale:
				graphsource[i,:] += dij
			else:
				graphsource[i,:] += dij

		# Right flux points
		for j in range(2):
			#cij = max(0.5*np.abs(rightcorrGrad[i]), np.abs(rightsolGrad[i]))
			cij = np.abs(nr[:,j]*rightcorrGrad[i])
			dij = cij*fluxStabTerm(ufr[j,:], u[i,:], lambdamethod)/detJ
			if pscale:
				graphsource[i,:] += dij
			else:
				graphsource[i,:] += dij

	return graphsource

# Variable sparsity graph source
def getVSGraphSource(eidx, nadj):
	global sol, nElems, p, detJ, nvars
	global solGradMat, leftcorrGrad, rightcorrGrad
	global lambdamethod, basiswts, graphupwind

	stride = p+1
	u = sol[stride*eidx:stride*(eidx+1),:]
	graphsource = np.zeros_like(u)

	[ul_ext, ul_int]  = commonSol('left', eidx)
	[ur_int, ur_ext]  = commonSol('right', eidx)

	# u at left/right flux points
	ufl = np.vstack((ul_ext, ul_int))
	ufr = np.vstack((ur_int, ur_ext))

	if graphupwind:
		nl = interpFluxSplit(ul_ext, ul_int)
		nr = -interpFluxSplit(ur_int, ur_ext)
	else:
		nl = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
		nr = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

	for i in range(p+1):
		GS = np.zeros((p+3, nvars))
		# Left flux points
		cij = leftcorrGrad[i]
		for j in range(2):
			du = ufl[j,:] - u[i,:]
			lmax = calcWaveSpeed(ufl[j,:] , u[i,:], lambdamethod)
			dij = np.abs(nl[:,j]*cij)*lmax/detJ
			if pscale:
				GS[0,:] += dij*du
			else:
				GS[0,:] += dij*du

		# Solution points
		for j in range(p+1):
			du = u[j,:] - u[i,:]
			lmax = calcWaveSpeed(u[j,:], u[i,:], lambdamethod)
			cij = solGradMat[i,j]
			dij = np.abs(cij)*lmax/detJ
			if pscale:
				GS[j+1,:] += dij*du*basiswts[i]
			else:
				GS[j+1,:] += dij*du

		# Right flux points
		cij = rightcorrGrad[i]
		for j in range(2):
			du = ufr[j,:] - u[i,:]
			lmax = calcWaveSpeed(ufr[j,:] , u[i,:], lambdamethod)
			dij = np.abs(nr[:,j]*cij)*lmax/detJ
			if pscale:
				GS[-1,:] += dij*du
			else:
				GS[-1,:] += dij*du

		startidx = max(0, i+1-nadj)
		endidx = min(p+3, i+1+nadj)
		graphsource[i,:] = np.sum(GS[startidx:endidx,:], axis=0)
	return graphsource

def DEBUGgetGraphSource(eidx):
	global sol, nElems, p, detJ
	global solGradMat, leftcorrGrad, rightcorrGrad 
	global lambdamethod, basiswts, fluxsplitmethod
	fluxsplitmethod = 'centered'

	stride = p+1
	u = sol[stride*eidx:stride*(eidx+1),:]
	graphsource = np.zeros_like(u)

	[ul_ext, ul_int]  = commonSol('left', eidx)
	[ur_int, ur_ext]  = commonSol('right', eidx)

	# u at left/right flux points
	ufl = np.vstack((ul_ext, ul_int))
	ufr = np.vstack((ur_int, ur_ext))

	#CALCULATE CIJ = DIV FLUX TERMS
	for i in range(p+1):
		for j in range(p+1):			
			cij = solGradMat[i,j]
			dij = cij*(calcFlux(u[j,:])[0])/detJ
			graphsource[i,:] += dij
		# Left flux points
		cij = leftcorrGrad[i]
		n = [0.5, -0.5]
		for j in range(2):
			dij = cij*(calcFlux(ufl[j,:])[0])/detJ
			graphsource[i,:] += n[j]*dij

		# Right flux points
		cij = rightcorrGrad[i]
		n = [-0.5, 0.5]
		for j in range(2):
			dij = cij*(calcFlux(ufr[j,:])[0])/detJ
			graphsource[i,:] += n[j]*dij

	'''
	# CALCULATE SUM CIJ = 0
	for i in range(p+1):
		for j in range(p+1):
			cij = solGradMat[i,j]
			graphsource[i,:] += cij
		# Left flux points
		cij = leftcorrGrad[i]
		n = [0.5, -0.5]
		for j in range(2):
			graphsource[i,:] += n[j]*cij

		# Right flux points
		cij = rightcorrGrad[i]
		n = [-0.5, 0.5]
		for j in range(2):
			graphsource[i,:] += n[j]*cij
		#graphsource[i,:] *= basiswts[i]
	'''


	return graphsource

def step(sol,dt):
	global timescheme
	if timescheme == 'rk4':
		k1 = -dt*divF(sol)
		k2 = -dt*divF(sol+k1/2.)
		k3 = -dt*divF(sol+k2/2.)
		k4 = -dt*divF(sol+k3)
		return sol + k1/6. + k2/3. + k3/3. + k4/6.
	elif timescheme == 'ssprk3':
		k1 = -dt*divF(sol)
		k2 = -dt*divF(sol+k1)
		k3 = -dt*divF(sol+k1/4.+k2/4.)
		return sol + k1/6. + k2/6. + 2.*k3/3.


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
		plt.plot(x, sol[:,0], '-')
		plt.xlabel('x')
		plt.ylabel('Density')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, GV-Upwind {4}, PScale {5}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, graphupwind, pscale))
		plt.grid()
		plt.figure(1)
		plt.plot(x, sol[:,1]/sol[:,0], '-')
		plt.xlabel('x')
		plt.ylabel('Velocity')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, GV-Upwind {4}, PScale {5}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, graphupwind, pscale))
		plt.grid()
		plt.figure(2)
		plt.plot(x, (gamma - 1.0)*(sol[:,2] - 0.5*(sol[:,1]**2)/sol[:,0]) , '-')
		plt.xlabel('x')
		plt.ylabel('Pressure')
		plt.title('P = {0}, DOF = {1}, t = {2} \n Flux {3}, GV-Upwind {4}, PScale {5}'.format(p, (p+1)*nElems, nt*dt, fluxsplitmethod, graphupwind, pscale))
		plt.grid()

meta = [p, dom, nt, nElems, dt, gamma, solptstype, fluxsplitmethod, lambdamethod, lbc, rbc]
with open(outpath, 'wb') as f:
	pickle.dump([meta, x, sol], f)

plt.show()




